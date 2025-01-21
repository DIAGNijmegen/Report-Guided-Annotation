from typing import Any, Callable, List, Optional, Tuple, Union

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
    softmax: "npt.NDArray[np.float64]",
    threshold: "float | np.floating[Any]" = 0.10,
    min_voxels_detection: int = 10,
    max_prob_round_decimals: Optional[int] = 4,
    lesion_confidence_aggregation: Callable[[np.ndarray], float] = np.max,
) -> "Tuple[npt.NDArray[np.float64], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """
    Extract lesion candidates from a softmax volume using a static threshold.
    """
    # load and preprocess softmax volume
    all_hard_blobs = np.zeros_like(softmax)
    confidences = []
    clipped_softmax = softmax.copy()
    clipped_softmax[softmax < threshold] = 0
    blobs_index, num_blobs = ndimage.label(clipped_softmax, structure=np.ones((3, 3, 3)))

    for idx in range(1, num_blobs+1):
        # determine mask for current lesion
        hard_mask = (blobs_index == idx).astype(int)

        if np.count_nonzero(hard_mask) <= min_voxels_detection:
            # remove small lesion candidates
            blobs_index[hard_mask.astype(bool)] = 0
            continue

        # add sufficiently sized detection
        hard_blob = hard_mask * clipped_softmax
        lesion_confidence = lesion_confidence_aggregation(hard_blob)
        if max_prob_round_decimals is not None:
            lesion_confidence = np.round(lesion_confidence, max_prob_round_decimals)
        hard_blob[hard_blob > 0] = lesion_confidence
        all_hard_blobs += hard_blob
        confidences.append((idx, lesion_confidence))
    return all_hard_blobs, confidences, blobs_index


def extract_lesion_candidates_dynamic(
    softmax: "npt.NDArray[np.float64]",
    min_voxels_detection: int = 10,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 2.5,
    max_prob_round_decimals: Optional[int] = None,
    remove_adjacent_lesion_candidates: bool = True,
    max_prob_failsafe_stopping_threshold: float = 0.01,
    version: int = 1,
    lesion_confidence_aggregation: Callable[[np.ndarray], float] = np.max,
) -> "Tuple[npt.NDArray[np.float64], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """
    Generate detection proposals using a dynamic threshold to determine the location and size of lesions.

    Version 1 contained a bug where the max. prob was not checked to be positive after removing small lesions.
    Version 2 fixes this bug.
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
            max_prob_round_decimals=max_prob_round_decimals,
            lesion_confidence_aggregation=lesion_confidence_aggregation,
        )

        # select blob with max. confidence
        # note: max_prob should be re-computed to account for the case where the max. prob
        # was inside a "lesion candidate" of less than min_voxels_detection, which is
        # thus removed in preprocess_softmax_static.
        max_prob = np.max(all_hard_blobs)

        if max_prob == 0 and version >= 2:
            # static lesion extraction only extracted small lesions. Remove them and continue
            mask_current_lesion = working_softmax > threshold
            working_softmax = (working_softmax * (~mask_current_lesion))
            continue

        # ensure that mask is only a single lesion candidate (this assumption fails when multiple lesions have the same max. prob)
        mask_current_lesion = (all_hard_blobs == max_prob)
        mask_current_lesion_indexed, _ = ndimage.label(mask_current_lesion, structure=np.ones((3, 3, 3)))
        mask_current_lesion = (mask_current_lesion_indexed == 1)

        # create mask with its confidence
        hard_blob = (all_hard_blobs * mask_current_lesion)

        # Detect whether the extractted mask is too close to an existing lesion candidate
        extracted_lesions_grown = ndimage.binary_dilation(dynamic_hard_blobs > 0, structure=np.ones((3, 3, 3)))
        current_lesion_has_overlap = (mask_current_lesion & extracted_lesions_grown).any()

        # Check if lesion candidate should be retained
        if remove_adjacent_lesion_candidates and current_lesion_has_overlap:
            # skip lesion candidate, as it is too close to an existing lesion candidate
            pass
        else:
            # store extracted lesion
            dynamic_hard_blobs += hard_blob
            confidences += [(tumor_index, float(max_prob))]
            dynamic_indexed_blobs += (mask_current_lesion * tumor_index)

        # remove extracted lesion from working-softmax
        working_softmax = (working_softmax * (~mask_current_lesion))

    return dynamic_hard_blobs, confidences, dynamic_indexed_blobs


def extract_lesion_candidates(
    softmax: "npt.NDArray[np.float64]",
    threshold: Union[str, float] = "dynamic-fast",
    min_voxels_detection: int = 10,
    num_lesions_to_extract: int = 5,
    dynamic_threshold_factor: float = 2.5,
    max_prob_round_decimals: Optional[int] = None,
    remove_adjacent_lesion_candidates: bool = True,
) -> "Tuple[npt.NDArray[np.float64], List[Tuple[int, float]], npt.NDArray[np.int_]]":
    """
    Generate detection proposals using a dynamic or static threshold to determine the size of lesions.

    Parameters
    ----------
    softmax : npt.NDArray[np.float64]
        Softmax prediction
    threshold : Union[str, float]
        Threshold to use for the extraction of lesion candidates.
        If "dynamic", multiple thresholds are used, based on the softmax volume.
        If "dynamic-fast", a single threshold is used, based on the softmax volume.
        If float, a static threshold is used (as specified).
    min_voxels_detection : int
        Minimum number of voxels in a lesion candidate.
    num_lesions_to_extract : int
        Number of lesion candidates to extract.
    dynamic_threshold_factor : float
        Ratio between max. of lesion candidate and its final extent. Higher factor means larger candidates.
    max_prob_round_decimals : Optional[int]
        Number of decimals to round the max. probability of a lesion candidate.
    remove_adjacent_lesion_candidates : bool
        If True, lesion candidates that are too close to an existing lesion candidate are removed.

    Returns
    -------
    hard_blobs : npt.NDArray[np.float64]
        Hard blobs of the input image, where each connected component is set to the lesion confidence.
    confidences : List[Tuple[int, float]]
        Confidences of the extracted lesion candidates (unordered).
        Each entry is a tuple of (is_lesion, lesion confidence).
    blobs_index : npt.NDArray[np.int_]
        Volume where each connected component is set to the index of the extracted lesion candidate.
    """
    # input validation
    if softmax.dtype in [np.dtype(np.float16)]:
        softmax = softmax.astype(np.float32)
    elif softmax.dtype in [np.dtype(np.longdouble)]:  # float128
        softmax = softmax.astype(np.float64)
    elif softmax.dtype in [np.dtype(np.csingle), np.dtype(np.cdouble), np.dtype(np.clongdouble)]:  # type: ignore[comparison-overlap]
        raise ValueError("Softmax predicitons should be of type float.")

    if threshold == "dynamic" or threshold == "dynamic-v2":
        version = 2 if "v2" in threshold else 1
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_dynamic(
            softmax=softmax,
            dynamic_threshold_factor=dynamic_threshold_factor,
            num_lesions_to_extract=num_lesions_to_extract,
            remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals,
            version=version,
        )
    elif threshold == "dynamic-mean" or threshold == "dynamic-v2-mean":
        version = 2 if "v2" in threshold else 1
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_dynamic(
            softmax=softmax,
            dynamic_threshold_factor=dynamic_threshold_factor,
            num_lesions_to_extract=num_lesions_to_extract,
            remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals,
            version=version,
            lesion_confidence_aggregation=np.mean,
        )
    elif threshold == "dynamic-fast":
        # determine max. softmax and set a per-case "static" threshold based on that
        max_prob = np.max(softmax)
        threshold = float(max_prob / dynamic_threshold_factor)
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_static(
            softmax=softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals,
        )
    elif threshold == "dynamic-fast-mean":
        # determine max. softmax and set a per-case "static" threshold based on that
        max_prob = np.max(softmax)
        threshold = float(max_prob / dynamic_threshold_factor)
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_static(
            softmax=softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals,
            lesion_confidence_aggregation=np.mean,
        )
    else:
        threshold = float(threshold)  # convert threshold to float, if it wasn't already
        all_hard_blobs, confidences, indexed_pred = extract_lesion_candidates_static(
            softmax=softmax,
            threshold=threshold,
            min_voxels_detection=min_voxels_detection,
            max_prob_round_decimals=max_prob_round_decimals,
        )

    return all_hard_blobs, confidences, indexed_pred
