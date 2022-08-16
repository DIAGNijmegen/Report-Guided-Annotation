import numpy as np
import pytest
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
