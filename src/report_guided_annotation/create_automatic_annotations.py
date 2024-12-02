import concurrent.futures
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from report_guided_annotation.extract_lesion_candidates import \
    extract_lesion_candidates

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


"""
Functions for automatically generating lesion annotations
Automatic annotations are created by fusing model predictions with
the number of csPCa lesions extracted from radiology reports.
Author: Joeran Bosma


References:
[1] J. S. Bosma, et. al. "Report Guided Automatic Lesion Annotation for Deep Learning Prostate Cancer Detection in bpMRI", to be submitted
[2] https://github.com/DIAGNijmegen/Report-Guided-Annotation
"""


def create_automatic_annotations_from_softmax(
    pred: "npt.NDArray[np.float64]",
    num_lesions_to_retain: int,
    threshold: str = 'dynamic'
) -> "Tuple[npt.NDArray[np.int_], npt.NDArray[np.float64], int]":
    """
    Create pseudo-labels from softmax prediction
    - pred: softmax predictions of shape (D, H, W) or (D, H, W, 2), with D, H and W spatial dimensions.
    - threshold: threshold for `extract_lesion_candidates` ('dynamic', 'dynamic-fast', 'otsu', or a static threshold)
    - num_lesions: number of lesions there should be (e.g., extracted from a radiology report)

    Returns:
    - hard automatic label (binary masks)
    - soft automatic label (softmax within mask)
    - number of lesions annotated
    """

    if len(pred.shape) == 4:
        # if softmax prediction is one-hot encoded, select the foreground channel
        assert pred.shape[-1] == 2, "Pseudo-label creation requires a softmax prediction of shape (D, H, W) or " + \
                                    f"(D, H, W, 2), with D, H and W spatial dimensions. Received shape: {pred.shape}"
        pred = pred[..., 1]

    """
    Step 1: Extract lesion candidates from softmax prediction
    Please refer to [1] or [2] for documentation on the lesion candidate extraction.
    This gives:
    - confidences: list of tuples with (lesion ID, lesion confidence), e.g. [(1, 0.2321), (2, 0.3453), (3, 0.0431), ...]
    - indexed_pred: numbered masks for the extracted lesion candidates. Lesion candidates are non-overlapping and the same shape as `pred`.
                    E.g., for the lesion candidate with confidence of 0.3453, each voxel has the value 2
    """
    # please note that the indexed predictions are in random order!
    _, confidences, indexed_pred = extract_lesion_candidates(
        pred,
        threshold=threshold,
        num_lesions_to_extract=num_lesions_to_retain,
    )

    """
    Step 2: Retain the n most confident lesions, with n the number of lesions to retain
    To keep memory usage low, this is performed in-place, by setting all lesion candidates we don't want to zero
    """
    # sort lesions based on confidence (from high confidence to low confidence)
    confidences = sorted(confidences, key=lambda idx_conf: idx_conf[1], reverse=True)

    # grab number of lesions to retain and select indices of lesions to remove
    idx_lesions_to_remove = [idx_conf[0] for idx_conf in confidences[num_lesions_to_retain:]]

    # set non-included lesions to zero
    for idx in idx_lesions_to_remove:
        mask = (indexed_pred == idx)
        indexed_pred[mask] = 0

    """
    Step 3: Create automatic labels
    Hard label: binarize the retained lesion masks
    Soft label: softmax predictions within the retained lesion masks
    """

    # binarize indexed mask to create hard automatic label
    lbl_hard = indexed_pred
    lbl_hard[lbl_hard > 0] = 1

    # mask softmax prediction with binary mask to create soft label
    lbl_soft = pred * lbl_hard

    # calculate the number of lesions in the automatic label (this can be lower than num_lesions_to_retain
    # when fewer candidates could be extracted from the softmax prediction)
    num_lesions_annotated = min(num_lesions_to_retain, len(confidences))

    return lbl_hard, lbl_soft, num_lesions_annotated


def create_automatic_annotations(
    prediction_map: "Dict[str, npt.NDArray[np.float64]]",
    num_lesions_to_retain_map: Dict[str, int],
    threshold: str = 'dynamic',
    skip_if_insufficient_lesions: bool = False,
    num_workers: int = 4,
    full_return: bool = False,
    verbose: bool = True,
    create_automatic_annotations_from_softmax_function=create_automatic_annotations_from_softmax,
) -> """Union[
        Tuple[Dict[str, npt.NDArray[np.int_]], Dict[str, npt.NDArray[np.float64]]],
        Tuple[Dict[str, npt.NDArray[np.int_]], Dict[str, npt.NDArray[np.float64]], List[str], List[str], List[str]],
]""":
    """
    Create automatic labels for multiple softmax predictions (with multiprocessing)
    The softmax predictions and number of lesions to retain should be specified as a dictionary:
    - prediction_map: {identifier1: pred1, identifier2: pred2, ...}
    - num_lesions_to_retain_map: {identifier1: num1, identifier2: num2, ...}

    Optional parameters:
    - threshold: threshold for `extract_lesion_candidates`, see above
    - skip_if_insufficient_lesions: whether to skip cases where fewer lesion candidates could be
                                    extracted than the target number of lesion specified by the user
    - num_workers: number of parallel calls to create automatic labels

    Returns:
    - dictionary of hard automatic labels: {identifier1: lbl1, identifier2: lbl2, ...}
    - dictionary of soft automatic labels: {identifier1: lbl1, identifier2: lbl2, ...}
    Optional:
    - cases_automatic_label_successful: list of identifiers for which automatic labels were generated
    - cases_num_lesions_to_retain_not_found: list of identifiers for which the number of lesions to retain was not found
    - cases_insufficient_lesions: list of identifiers for which the number of lesion candidates extracted from the softmax prediction
                                  was lower than the target number of lesions to retain (if skip_if_insufficient_lesions=True)
    """

    # create placeholders for automatic labels
    pseudo_labels_hard = {}
    pseudo_labels_soft = {}

    # keep track of which cases failed
    cases_num_lesions_to_retain_not_found = []
    cases_insufficient_lesions = []
    future_to_args = {}

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for subject_id, pred in prediction_map.items():
            # check if target number of lesions is specified
            if subject_id not in num_lesions_to_retain_map:
                cases_num_lesions_to_retain_not_found += [subject_id]
                continue

            # add creation of automatic labels to queue
            future = pool.submit(
                create_automatic_annotations_from_softmax_function,
                pred=pred,
                num_lesions_to_retain=num_lesions_to_retain_map[subject_id],
                threshold=threshold,
            )
            future_to_args[future] = subject_id

        if len(cases_num_lesions_to_retain_not_found) > 0:
            print(f"Did not find number of lesions to retain for {len(cases_num_lesions_to_retain_not_found)} cases, skipped those.")

        # process cases in parallel
        iterator = concurrent.futures.as_completed(future_to_args)
        if verbose:
            iterator = tqdm(iterator, total=len(future_to_args))
        for future in iterator:
            subject_id = future_to_args[future]
            try:
                lbl_hard, lbl_soft, num_lesions_annotated = future.result()
            except Exception as e:
                print(f"Exception: {e}")
            else:
                if skip_if_insufficient_lesions and (num_lesions_to_retain_map[subject_id] > num_lesions_annotated):
                    cases_insufficient_lesions += [subject_id]
                    continue

                # collect automatic labels
                pseudo_labels_hard[subject_id] = lbl_hard
                pseudo_labels_soft[subject_id] = lbl_soft

    if len(cases_insufficient_lesions) > 0:
        print(f"Have fewer lesion candidates than target number of lesions for {len(cases_insufficient_lesions)} cases, skipped those.")

    # return results
    if full_return:
        cases_automatic_label_successful = list(pseudo_labels_hard)
        return pseudo_labels_hard, pseudo_labels_soft, cases_automatic_label_successful, cases_num_lesions_to_retain_not_found, cases_insufficient_lesions

    return pseudo_labels_hard, pseudo_labels_soft


def create_automatic_annotations_for_folder(
    input_dir: str,
    output_dir: str,
    num_lesions_to_retain_map_path: Optional[str] = None,
    **kwargs
) -> "Union[int, Tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]]":
    """
    Create automatic labels for multiple softmax predictions (with multiprocessing)
    The softmax predictions should be individual .nii.gz/.npz/.npy files in the input dictionary:

    predictions
    ├── ProstateX-0000.nii.gz
    ├── ProstateX-0001.nii.gz
    ├── ProstateX-0002.nii.gz
    ...


    The number of lesions to retain should be stored as a json dictionary, with the filenames of the
    predictions that should be converted to automatic annotations as keys, and the number of lesions
    to retain as values (here the PI-RADS >= 3 lesions are retained).
    {
        "ProstateX-0000.nii.gz": 1,
        "ProstateX-0001.nii.gz": 1,
        "ProstateX-0002.nii.gz": 2,
        ...
    }


    Optional parameters:
    - threshold: threshold for `extract_lesion_candidates`, see above
    - skip_if_insufficient_lesions: whether to skip cases where fewer lesion candidates could be
                                    extracted than the target number of lesion specified by the user
    - num_workers: number of parallel calls to create automatic labels

    Returns:
    - dictionary of hard automatic labels: {identifier1: lbl1, identifier2: lbl2, ...}
    - dictionary of soft automatic labels: {identifier1: lbl1, identifier2: lbl2, ...}
    Optional:
    - cases_automatic_label_successful: list of identifiers for which automatic labels were generated
    - cases_num_lesions_to_retain_not_found: list of identifiers for which the number of lesions to retain was not found
    - cases_insufficient_lesions: list of identifiers for which the number of lesion candidates extracted from the softmax prediction
                                  was lower than the target number of lesions to retain (if skip_if_insufficient_lesions=True)
    """
    # read number of lesion candidates to retain
    if num_lesions_to_retain_map_path is None:
        num_lesions_to_retain_map_path = os.path.join(input_dir, "num_lesions_to_retain_map.json")
    with open(num_lesions_to_retain_map_path) as fp:
        num_lesions_to_retain_map = json.load(fp)

    print(f"Found {len(num_lesions_to_retain_map)} samples in num_lesions_to_retain_map.json")
    print(f"Here are some examples, please check if they look okay: \n{list(num_lesions_to_retain_map)[0:5]}\n")

    # read predictions
    for pred_fn in tqdm(num_lesions_to_retain_map, desc='Creating automatic annotations'):
        pred, pred_itk = None, None
        pred_path = os.path.join(input_dir, pred_fn)
        pred_fn_out = pred_fn.replace('.npy', '.nii.gz').replace('.npz', '.nii.gz')
        automatic_annotation_path = os.path.join(output_dir, pred_fn_out)
        if os.path.exists(automatic_annotation_path):
            print(f"Skipping {pred_fn_out} because it already exists.")
            continue

        if ('.nii.gz' in pred_fn) or ('.mha' in pred_fn) or ('.mhd' in pred_fn) or ('.nii' in pred_fn):
            pred_itk = sitk.ReadImage(pred_path)  # any format supported by SimpleITK
            pred = sitk.GetArrayFromImage(pred_itk)
        elif '.npy' in pred_fn:
            pred = np.load(pred_path)  # numpy array
        elif '.npz' in pred_fn:
            pred = np.load(pred_path)['softmax'].astype('float32')[1]  # nnUnet format
        else:
            raise ValueError(f"Unsupported file extension for {pred_fn}! Available are: "
                             ".nii.gz, .nii, .mha, .mhd, .npy and .npz (nnUNet format).")

        # create automatic annotation
        pseudo_labels_hard, *_ = create_automatic_annotations(
            prediction_map={pred_fn: pred},  # type: ignore
            num_lesions_to_retain_map=num_lesions_to_retain_map,
            verbose=False,
            **kwargs
        )

        if pred_fn not in pseudo_labels_hard:
            # Could not create automatic annotation for this case, did not find sufficient lesion candidates
            # (warning message is displayed in create_automatic_annotations)
            continue

        # write psuedo labels to output directory
        # construct target filename (same as input, unless it is numpy/compressed numpy)
        write_lbl(
            lbl=pseudo_labels_hard[pred_fn],
            dst_path=automatic_annotation_path,
            reference_img=pred_itk,
            create_parent_folder=True,
        )

    print(f"Finished creating automatic annotations, see output folder {output_dir}. ")
    return 1


def write_lbl(
    lbl: "npt.NDArray[np.int_]",
    dst_path: str,
    reference_img: "Optional[sitk.Image]" = None,
    create_parent_folder: bool = True
) -> None:
    """Write automatic annotation to nifti"""
    lbl_itk = sitk.GetImageFromArray(lbl.astype(np.int8), sitk.sitkInt8)

    if reference_img is not None:
        lbl_itk.CopyInformation(reference_img)

    if create_parent_folder:
        # ensure target folder exists
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

    sitk.WriteImage(lbl_itk, dst_path, useCompression=True)
