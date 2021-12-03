import os

from report_guided_annotation import extract_pirads_scores


def test_parse_report():
    # define scores found in the radiology reports (verified manually)
    expected_results = {
        'sample_1.txt': [
            (1, {'T2W': '3', 'DWI': '5', 'DCE': '+', 'tot': 5}),
            (2, {'T2W': '2', 'DWI': '2', 'DCE': '+', 'tot': 2})
        ],
        'sample_2.txt': [
            (1, {'T2W': '5', 'DWI': '5', 'DCE': '+', 'tot': 5}),
            (2, {'T2W': '5', 'DWI': '5', 'DCE': '+', 'tot': 5})
        ],
        'sample_3.txt': [
            (1, {'T2W': '4', 'DWI': '4', 'DCE': '+', 'tot': 4}),
        ],
    }

    for report_fn in expected_results.keys():
        # read radiology report
        with open(os.path.join("tests/reports", report_fn)) as fp:
            report = fp.read()

        # parse report and extract scores
        res = extract_pirads_scores(
            report,
            subject_id=report_fn,
            aggressive=True,
            flatten_report=True,
            ignore_conclusion=False,
            conclusion_fallback=True,
            conclusion_only=False,
            strict=False,
            conclusion_fallback_missing_pirads=False,
            verbose=1,
        )

        # verify extracted scores
        expected_res = expected_results[report_fn]
        for (expected_lesion_id, expected_scores), (lesion_id, scores) in zip(expected_res, res):
            assert expected_lesion_id == lesion_id
            for key, score in expected_scores.items():
                assert score == scores[key]
