import argparse

from report_guided_annotation import create_automatic_annotations_for_folder

"""
Create Report Guided Lesion Annotations from softmax model predictions
Author: Joeran Bosma

https://github.com/DIAGNijmegen/Report-Guided-Annotation
"""

# parse input and output folders
parser = argparse.ArgumentParser(description='Command line options')
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Path to folder with softmax model predicitons")
parser.add_argument("-o", "--output", type=str, required=True,
                    help="Path to folder to store automatic annotations")
parser.add_argument("-t", "--threshold", type=str, default="dynamic",
                    help=("Threshold to use for lesion candidate extraction: " +
                          "'dynamic', 'dynamic-fast' or a static float-threshold"))
parser.add_argument("-s", "--skip_if_insufficient_lesions", type=int, default=1,
                    help=("Whether to skip saving an annotation if insufficient " +
                          "lesion candidates can be extracted from the softmax " +
                          "model predictions (default: True)"))
args = parser.parse_args()

print(f"""
    Creating Report Guided Annotations
    Input folder: {args.input}
    Output folder: {args.output}
""")

create_automatic_annotations_for_folder(
    input_dir=args.input,
    output_dir=args.output,
    threshold=args.threshold,
    skip_if_insufficient_lesions=args.skip_if_insufficient_lesions,
    num_workers=1
)
