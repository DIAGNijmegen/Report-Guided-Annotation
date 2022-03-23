import os
import json
import numpy as np
from pathlib import Path
from scipy import ndimage
import SimpleITK as sitk

from report_guided_annotation import create_automatic_annotations_for_folder


def test_create_automatic_labels_for_folder():
    # define input and output folders
    input_dir = Path("tests/input/")
    output_dir = Path("tests/output/")

    # generate report-guided annotations
    create_automatic_annotations_for_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold='dynamic',
        skip_if_insufficient_lesions=True,
        num_workers=4,
    )

    # verify generated annotations exist
    assert os.path.exists(output_dir / "ProstateX-0000_07-07-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-05711.nii.gz")

    # read number of lesions there should be
    with open(input_dir / "num_lesions_to_retain_map.json") as fp:
        num_lesions_to_retain_map = json.load(fp)

    # for each case, check the number of lesions
    for subject_fn, num_lesions_to_retain in num_lesions_to_retain_map.items():
        # read automatic annotation
        automatic_annot = sitk.ReadImage(str(output_dir / subject_fn))
        automatic_annot = sitk.GetArrayFromImage(automatic_annot)
        _, num_blobs = ndimage.label(automatic_annot, structure=np.ones((3, 3, 3)))
        assert num_lesions_to_retain == num_blobs
