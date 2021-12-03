import os
import json
from report_guided_annotation import extract_pirads_scores


def test_example_script():
    path_to_reports = "tests/reports"
    path_to_softmax = "tests/parse_reports_output"

    num_lesions_to_retain_map = {}

    for fn in os.listdir(path_to_reports):
        if '.txt' not in fn:
            print(f"Skipped {fn}, not a repot?")
            continue

        # read radiology report
        with open(os.path.join(path_to_reports, fn)) as fp:
            report = fp.read()

        # extract PI-RADS scores from radiology report
        scores = extract_pirads_scores(
            report=report
        )

        # count number of PI-RADS >= 4 lesions
        pirads_scores = [int(lesion_scores['tot'])
                         for (lesion_id, lesion_scores) in scores
                         if lesion_scores['tot'] is not None]
        num_pirads_45 = sum([score >= 4 for score in pirads_scores])

        # store number of clinically significant report findings
        softmax_fn = fn.replace(".txt", ".nii.gz")
        num_lesions_to_retain_map[softmax_fn] = num_pirads_45

    # save number of lesion candidates to retain
    with open(os.path.join(path_to_softmax, "num_lesions_to_retain_map.json"), "w") as fp:
        json.dump(num_lesions_to_retain_map, fp, indent=4)

    with open(os.path.join(path_to_softmax, "num_lesions_to_retain_map.json")) as fp:
        num_lesions_to_retain_map = json.load(fp)

    assert num_lesions_to_retain_map == {
        "sample_1.nii.gz": 1,
        "sample_2.nii.gz": 2,
        "sample_3.nii.gz": 1
    }
