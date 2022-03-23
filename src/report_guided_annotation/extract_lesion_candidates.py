import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import itertools
import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional, Union, Iterable, Sequence, Hashable, Dict, Any
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

"""
Extract lesion candidates from a softmax prediction
Authors: anindox8, matinhz, joeranbosma
"""


# Preprocess Softmax Volume (Clipping, Max Confidence)
def preprocess_softmax_static(
    softmax: "npt.NDArray[np.float_]",
    threshold: float = 0.10,
    min_voxels_detection: int = 10,
    max_prob_round_decimals: Optional[int] = 4
) -> "Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    # Load and Preprocess Softmax Image
    all_hard_blobs = np.zeros_like(softmax)
    confidences = []
    clipped_softmax = softmax.copy()
    clipped_softmax[softmax < threshold] = 0
    blobs_index, num_blobs = ndimage.label(clipped_softmax, structure=np.ones((3, 3, 3)))

    if num_blobs > 0:  # For Each Prediction
        for tumor in range(1, num_blobs+1):
            # determine mask for current lesion
            hard_mask = np.zeros_like(blobs_index)
            hard_mask[blobs_index == tumor] = 1

            if np.count_nonzero(hard_mask) <= min_voxels_detection:
                # remove tiny detection of <= 0.009 cm^3
                blobs_index[hard_mask.astype(bool)] = 0
                continue

            # add sufficiently sized detection
            hard_blob = hard_mask * clipped_softmax
            max_prob = np.max(hard_blob)
            if max_prob_round_decimals is not None:
                max_prob = np.round(max_prob, max_prob_round_decimals)
            hard_blob[hard_blob > 0] = max_prob
            all_hard_blobs += hard_blob
            confidences.append((tumor, max_prob))
    return all_hard_blobs, confidences, blobs_index


def preprocess_softmax_dynamic(
    softmax: "npt.NDArray[np.float_]",
    min_voxels_detection: int = 10,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 2.5,
    max_prob_round_decimals: Optional[int] = None,
    remove_adjacent_lesion_candidates: bool = True,
    max_prob_failsafe_stopping_threshold: float = 0.01
) -> "Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """
    Generate detection proposals using a dynamic threshold to determine the location and size of lesions.
    Author: Joeran Bosma
    """
    working_softmax = softmax.copy()
    dynamic_hard_blobs = np.zeros_like(softmax)
    confidences: List[Tuple[int, float]] = []
    dynamic_indexed_blobs: "npt.NDArray[np.int_]" = np.zeros_like(softmax, dtype=int)

    while len(confidences) < num_lesions_to_extract:
        tumor_index = 1 + len(confidences)

        # determine max. softmax
        max_prob = np.max(working_softmax)

        if max_prob < max_prob_failsafe_stopping_threshold:
            break

        # set dynamic threshold to half the max
        threshold = max_prob / dynamic_threshold_factor

        # extract blobs for dynamix threshold
        all_hard_blobs, _, _ = preprocess_softmax_static(working_softmax, threshold=threshold,
                                                         min_voxels_detection=min_voxels_detection,
                                                         max_prob_round_decimals=max_prob_round_decimals)

        # select blob with max. confidence
        # note: the max_prob is re-computed in the (unlikely) case that the max. prob
        # was inside a 'lesion candidate' of less than min_voxels_detection, which is
        # thus removed in preprocess_softmax_static.
        max_prob = np.max(all_hard_blobs)
        mask_current_lesion = (all_hard_blobs == max_prob)

        # ensure that mask is only a single lesion candidate (this assumption fails when multiple lesions have the same max. prob)
        mask_current_lesion_indexed, _ = ndimage.label(mask_current_lesion, structure=np.ones((3, 3, 3)))
        mask_current_lesion = (mask_current_lesion_indexed == 1)

        # create mask with its confidence
        hard_blob = (all_hard_blobs * mask_current_lesion)

        # Detect whether the extractted mask is a ring/hollow sphere
        # around an existing lesion candidate. For confident lesions,
        # the surroundings of the prediction are still quite confident,
        # and can become a second 'detection'. For an # example, please
        # see extracted lesion candidates nr. 4 and 5 at:
        # https://repos.diagnijmegen.nl/trac/ticket/9299#comment:49
        # Detection method: grow currently extracted lesions by one voxel,
        # and check if they overlap with the current extracted lesion.
        extracted_lesions_grown = ndimage.morphology.binary_dilation(dynamic_hard_blobs > 0)
        current_lesion_has_overlap = (mask_current_lesion & extracted_lesions_grown).any()

        # Check if lesion candidate should be retained
        if (not remove_adjacent_lesion_candidates) or (not current_lesion_has_overlap):
            # store extracted lesion
            dynamic_hard_blobs += hard_blob
            confidences += [(tumor_index, max_prob)]
            dynamic_indexed_blobs += (mask_current_lesion * tumor_index)

        # remove extracted lesion from working-softmax
        working_softmax = (working_softmax * (~mask_current_lesion))

    return dynamic_hard_blobs, confidences, dynamic_indexed_blobs


def preprocess_softmax(
    softmax: "npt.NDArray[np.float_]",
    threshold: Union[str, float] = 'dynamic-fast',
    min_voxels_detection: int = 10,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 2.5,
    max_prob_round_decimals: Optional[int] = None,
    remove_adjacent_lesion_candidates: bool = True,
) -> "Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """
    Generate detection proposals using a dynamic or static threshold to determine the size of lesions.
    """
    if threshold == 'dynamic':
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_dynamic(softmax, min_voxels_detection=min_voxels_detection,
                                                                               dynamic_threshold_factor=dynamic_threshold_factor,
                                                                               num_lesions_to_extract=num_lesions_to_extract,
                                                                               remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates,
                                                                               max_prob_round_decimals=max_prob_round_decimals)
    elif threshold == 'dynamic-fast':
        # determine max. softmax and set a per-case 'static' threshold based on that
        max_prob = np.max(softmax)
        threshold = float(max_prob / dynamic_threshold_factor)
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_static(softmax, threshold=threshold,
                                                                              min_voxels_detection=min_voxels_detection,
                                                                              max_prob_round_decimals=max_prob_round_decimals)
    else:
        threshold = float(threshold)  # convert threshold to float, if it wasn't already
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_static(softmax, threshold=threshold,
                                                                              min_voxels_detection=min_voxels_detection,
                                                                              max_prob_round_decimals=max_prob_round_decimals)

    return all_hard_blobs, confidences, indexed_pred


def validate_and_convert(*inputs) -> "List[npt.NDArray[Any]]":
    """
    Validate inputs:
    - If the inputs consists of at least one dictionary, all inputs should be a dictionary
    - If the inputs are dictionaries, check if the keys are the same

    Convert inputs:
    - If the inputs are dictionaries, reduce to flat lists, with the same order for each input
    - Convert to numpy arrays
    """
    # check if any of the inputs is a dictionary
    any_dict = False
    for inp in inputs:
        if isinstance(inp, dict):
            any_dict = True

    if any_dict:
        # check if all inputs are dictionaries
        assert all(isinstance(inp, dict) for inp in inputs), (
            "Inputs must either all be dictionaries, or all "
            "iterables (with cases in the same order)!"
        )

        # check if all cases are present in each dictionary
        case_ids = set(inputs[0])
        assert all(case_ids == set(inp) for inp in inputs), \
            "Inputs must all contain the same cases!"

        # collect flat lists with cases in the same order
        ordered_case_ids = sorted(list(case_ids))
        inputs_flat = ([inp[c] for c in ordered_case_ids] for inp in inputs)

        # convert to numpy arrays
        return [np.array(inp) for inp in inputs_flat]

    # convert to numpy arrays
    return [np.array(inp) for inp in inputs]


def preprocess_softmaxes(
    y_pred: "Union[Dict[Hashable, npt.NDArray[np.float_]], Sequence[npt.NDArray[np.float_]], npt.NDArray[np.float_]]",
    subject_list: Optional[Iterable[Hashable]] = None,
    threshold: Union[str, float] = 'dynamic-fast',
    min_voxels_detection: int = 10,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 2.5,
    max_prob_round_decimals: Optional[int] = None,
    remove_adjacent_lesion_candidates: bool = True,
    max_workers: Optional[int] = None,
    flat: Optional[bool] = None,
    verbose: int = 1,
) -> "Dict[Hashable, Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]]":
    """
    Preprocess all softmax volumes using multiprocessing
    """
    if subject_list is None:
        # generate indices to keep track of each case during multiprocessing
        subject_list = itertools.count()
        if flat is None:
            flat = True

    # placeholders
    future_to_args = {}
    results: "Dict[Hashable, Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]]" = {}

    # input validation
    y_pred = validate_and_convert(*y_pred)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for subject_id, softmax in zip(subject_list, y_pred):
            # add lesion extraction to the queue
            future = pool.submit(
                preprocess_softmax,
                softmax=softmax, threshold=threshold, min_voxels_detection=min_voxels_detection,
                num_lesions_to_extract=num_lesions_to_extract, dynamic_threshold_factor=dynamic_threshold_factor,
                max_prob_round_decimals=max_prob_round_decimals, remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates,
            )
            future_to_args[future] = subject_id

    # process cases in parallel
    iterator = concurrent.futures.as_completed(future_to_args)
    if verbose:
        iterator = tqdm(iterator, total=len(future_to_args))
    for future in iterator:
        subject_id = future_to_args[future]
        try:
            all_hard_blobs, confidences, indexed_pred = future.result()
            results[subject_id] = (all_hard_blobs, confidences, indexed_pred)
        except Exception as e:
            print(f"Exception: {e} for {subject_id}")

    return results
