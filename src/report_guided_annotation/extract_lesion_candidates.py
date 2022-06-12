from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

"""
Extract lesion candidates from a softmax prediction
Authors: anindox8, matinhz, joeranbosma
"""


def extract_lesion_candidates_static(
    softmax: "npt.NDArray[np.float_]",
    threshold: float = 0.10,
    min_voxels_detection: int = 10,
    max_prob_round_decimals: Optional[int] = 4
) -> "Tuple[npt.NDArray[np.float_], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """Extract lesion candidates from a softmax volume"""
    # load and preprocess softmax volume
    all_hard_blobs = np.zeros_like(softmax)
    confidences = []
    clipped_softmax = softmax.copy()
    clipped_softmax[softmax < threshold] = 0
    blobs_index, num_blobs = ndimage.label(clipped_softmax, structure=np.ones((3, 3, 3)))

    for idx in range(1, num_blobs+1):
        # determine mask for current lesion
        hard_mask = np.zeros_like(blobs_index)
        hard_mask[blobs_index == idx] = 1

        if np.count_nonzero(hard_mask) <= min_voxels_detection:
            # remove small lesion candidates
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        # add sufficiently sized detection
        hard_blob = hard_mask * clipped_softmax
        max_prob = np.max(hard_blob)
        if max_prob_round_decimals is not None:
            max_prob = np.round(max_prob, max_prob_round_decimals)
        hard_blob[hard_blob > 0] = max_prob
        all_hard_blobs += hard_blob
        confidences.append((idx, max_prob))
    return all_hard_blobs, confidences, blobs_index


def extract_lesion_candidates_dynamic(
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
        all_hard_blobs, _, _ = extract_lesion_candidates_static(
            softmax=working_softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals
        )

        # select blob with max. confidence
        # note: max_prob should be re-computed to account for the case where the max. prob
        # was inside a 'lesion candidate' of less than min_voxels_detection, which is
        # thus removed in preprocess_softmax_static.
        max_prob = np.max(all_hard_blobs)
        mask_current_lesion = (all_hard_blobs == max_prob)

        # ensure that mask is only a single lesion candidate (this assumption fails when multiple lesions have the same max. prob)
        mask_current_lesion_indexed, _ = ndimage.label(mask_current_lesion, structure=np.ones((3, 3, 3)))
        mask_current_lesion = (mask_current_lesion_indexed == 1)

        # create mask with its confidence
        hard_blob = (all_hard_blobs * mask_current_lesion)

        # Detect whether the extractted mask is too close to an existing lesion candidate
        extracted_lesions_grown = ndimage.morphology.binary_dilation(dynamic_hard_blobs > 0, structure=np.ones((3, 3, 3)))
        current_lesion_has_overlap = (mask_current_lesion & extracted_lesions_grown).any()

        # Check if lesion candidate should be retained
        if remove_adjacent_lesion_candidates and current_lesion_has_overlap:
            # skip lesion candidate, as it is too close to an existing lesion candidate
            pass
        else:
            # store extracted lesion
            dynamic_hard_blobs += hard_blob
            confidences += [(tumor_index, max_prob)]
            dynamic_indexed_blobs += (mask_current_lesion * tumor_index)

        # remove extracted lesion from working-softmax
        working_softmax = (working_softmax * (~mask_current_lesion))

    return dynamic_hard_blobs, confidences, dynamic_indexed_blobs


def extract_lesion_candidates(
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
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_dynamic(
            softmax=softmax,
            dynamic_threshold_factor=dynamic_threshold_factor,
            num_lesions_to_extract=num_lesions_to_extract,
            remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals
        )
    elif threshold == 'dynamic-fast':
        # determine max. softmax and set a per-case 'static' threshold based on that
        max_prob = np.max(softmax)
        threshold = float(max_prob / dynamic_threshold_factor)
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_static(
            softmax=softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals
        )
    else:
        threshold = float(threshold)  # convert threshold to float, if it wasn't already
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_static(
            softmax=softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals
        )

    return all_hard_blobs, confidences, indexed_pred
