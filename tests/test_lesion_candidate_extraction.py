import numpy as np
import pytest
import SimpleITK as sitk

from report_guided_annotation import extract_lesion_candidates


@pytest.mark.parametrize("dtype", [
    np.bool_,
    np.byte,
    np.ubyte,
    np.short,
    np.ushort,
    np.intc,
    np.uintc,
    np.int_,
    np.uint,
    np.longlong,
    np.ulonglong,
    np.float16,
    np.single,
    np.double,
    np.longdouble,
])
def test_lesion_candidate_extraction_dtypes(dtype):
    softmax = np.zeros((3, 3, 3), dtype=dtype)
    extract_lesion_candidates(softmax)


@pytest.mark.xfail
@pytest.mark.parametrize("dtype", [
    np.csingle,
    np.cdouble,
    np.clongdouble,
])
def test_lesion_candidate_extraction_dtypes_xfail(dtype):
    softmax = np.zeros((3, 3, 3), dtype=dtype)
    extract_lesion_candidates(softmax)


def test_lesion_candidate_extraction_dynamic_edge_case():
    """
    Test the dynamic lesion extraction for an edge case.
    The extraction of lesion candidates results in two tiny
    "islands" of lesion candidates. These are removed due
    to the min_voxels_detection=10. Subsequently, the
    dynamic threshold is too high to detect any lesion
    candidates. This results in an empty output.
    The new version of the dynamic lesion extraction
    should not fail in this case.
    """

    # load prediction
    pred = sitk.ReadImage('tests/input/dynamic_extraction_edge_case.mha')
    pred = sitk.GetArrayFromImage(pred)

    # extract lesion candidates
    hard_blobs, confidences, blobs_index = extract_lesion_candidates(
        softmax=pred,
        threshold='dynamic-v2',
    )

    # check output
    assert len(confidences) == 5, "Expected 5 lesion candidates."
