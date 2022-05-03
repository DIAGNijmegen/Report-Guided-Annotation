from __future__ import absolute_import
from report_guided_annotation.create_automatic_annotations import (
      create_automatic_annotations_from_softmax,
      create_automatic_annotations,
      create_automatic_annotations_for_folder,
)
from report_guided_annotation.parse_report import extract_pirads_scores
from report_guided_annotation.extract_lesion_candidates import extract_lesion_candidates

print("\n\nPlease cite the following paper when using Report Guided Annotations:\n\nBosma, J.S., et al. "
      "\"Semi-supervised learning with report-guided lesion annotation for deep learning-based prostate cancer detection in bpMRI\" "
      "to be submitted\n\n")
print("If you have questions or suggestions, feel free to open an issue at https://github.com/DIAGNijmegen/Report-Guided-Annotation\n")

__all__ = [
      # explicitly expose these functions
      "create_automatic_annotations_from_softmax",
      "create_automatic_annotations",
      "create_automatic_annotations_for_folder",
      "extract_pirads_scores",
      "extract_lesion_candidates",
]
