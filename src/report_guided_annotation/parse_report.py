import numpy as np
import re

from typing import Optional, List, Tuple, Dict, Union


def extract_pirads_scores(
    report: str,
    ignore_addendum: bool = True,
    subject_id: Optional[str] = None,
    aggressive: bool = False,
    conclusion_fallback: bool = False,
    conclusion_only: bool = False,
    strict: bool = False,
    conclusion_fallback_missing_pirads: bool = False,
    ignore_conclusion: bool = False,
    flatten_report: bool = False,
    verbose: int = 1
) -> List[Tuple[int, Dict[str, Union[str, int, None]]]]:
    """
    Extract PI-RADS scores from radiology report.

    Input:
    Radiology report for prostate cancer detection, reported with PI-RADS v2

    Mechanism:
    Most of the radiology reports in our dataset were generated from a template, and
    modified to provide additional information. Although multiple templates were used
    over the years, this resulted in structured reports for most visits. This makes a
    rule-based natural language processing script a reliable and transparent way to
    extract PI-RADS scores from our radiology reports.

    Simply counting the occurrences of `PI-RADS 4/5` in the report body is reasonably
    effective, but has some pitfalls. For example, prior PI-RADS scores are often
    referenced during follow-up visits, resulting in false positive matches. Findings
    can also be grouped and described jointly, resulting in false negatives. To improve
    the reliability of the PI-RADS extraction from radiology reports, we extract the
    scores in two steps.

    First, we try to split the radiology reports in sections for individual findings,
    by searching for text that matches the following structure:
    ```
    [Finding] (number indicator) [number]
    ```
    Where `Finding` matches the Dutch translations `afwijking`, `laesie`, `markering`
    or `regio`. The optional number indicators are `nr.`, `mark` and `nummer`. The
    number at the end matches one or multiple numbers (e.g., `1` or `2+3`).

    Secondly, we extract the PI-RADS scores by searching for text that matches the
    following structure:
    ```
    [PI-RADS] (separators) [number 1-5]
    ```
    Where the optional separators include `v2 category` and `:`. The dash between `PI`
    and `RADS' is optional. The T2W, DWI and DCE scores are extracted analogous to the
    PI-RADS score, while also allowing joint extraction:
    ```
    T2W/DWI/DCE score: [1-5]/[1-5]/[-+]
    ```
    In this instance, the first number is matched with the T2W score, the second with
    DWI and the `+` or `-` with DCE.

    In case the report could not be split in sections per lesion, we apply strict pattern
    matching on the full report. During strict pattern matching we only extract T2W, DWI
    and DCE scores jointly, to ensure the scores are from the same lesion. The resulting
     PI-RADS scores are extracted from the full report and matched to the individual scores.

    Example report sections with the text matching the PI-RADS, T2W, DWI and DCE scores
    coloured in are shown in J. S. Bosma, et. al. "Report Guided Automatic Lesion Annotation
    for Deep Learning Prostate Cancer Detection in bpMRI", to be submitted, Figure 2.

    Returns:
    - a list of tuples, where each tuple contains the lesion id and PI-RADS scores for
    that lesion. For example:
    [
        (1, {'T2W': '3', 'DWI': '5', 'DCE': '+', 'tot': 5}),
        (2, {'T2W': '2', 'DWI': '2', 'DCE': '+', 'tot': 2})
    ]
    """

    original_report = report

    if ignore_addendum:
        report = remove_addendum_from_report(report)

    if ignore_conclusion:
        report = remove_conclusion_from_report(report)

    if flatten_report:
        report = report.replace("\n", " ")
        report = report.replace("  ", " ")

    results: List[Tuple[int, Dict[str, Union[str, int, None]]]] = []
    extraction_type = -1

    if not conclusion_only:
        report_snippets = extract_lesion_sections(report, verbose=verbose)
        report_split_in_snippets = None

        if report_snippets is None:
            raise Exception(f"No match for report {report}")
        else:
            if report_snippets[0][0] == 0:
                print("Report not splitted in snippets, working on full report") if verbose >= 2 else None
                report_split_in_snippets = False
            else:
                print(f"Found {len(report_snippets)} report snippets") if verbose >= 2 else None
                report_split_in_snippets = True

        # extract scores
        for lesion_number, section_text in report_snippets:
            if lesion_number > 0:
                assert report_split_in_snippets
                # found numbered lesion sections
                scores_lesion = extract_pirads_scores_from_lesion_section(section_text, verbose=verbose)
                results += [(lesion_number, scores_lesion)]
            else:
                assert not report_split_in_snippets and lesion_number == 0
                # did not found numbered sections, look for multiple scores in the same section
                scores = extract_all_scores_from_full_report(report, subject_id=subject_id, strict=strict,
                                                             aggressive=aggressive, verbose=verbose)
                results += scores

        # determine extraction type (up to this point)
        if report_split_in_snippets:
            if len(results):
                # check if all results have final PI-RADS
                if any([scores['tot'] is None or np.isnan(float(scores['tot'])) for i, scores in results]):
                    extraction_type = 0
                else:
                    extraction_type = 1
            else:
                extraction_type = 2
        else:
            if len(results):
                extraction_type = 3
            else:
                extraction_type = 4

    empty_or_nan = (len(results) == 0) or any([s['tot'] is None or np.isnan(float(s['tot'])) for i, s in results])
    if (not results and conclusion_fallback) or conclusion_only or (empty_or_nan and conclusion_fallback_missing_pirads):
        # search for PI-RADS scores in the conclusion
        if 'Impressie:' in original_report:
            conclusion = original_report.split("Impressie:")[-1]

            # extract total scores
            for pattern in [
                fr'PI-?RADS v2 categorie{allowed_separators}{PIRADS_pattern}',
                fr'PI-?RADS v2 category{allowed_separators}{PIRADS_pattern}',
                fr'PI-?RADS{allowed_separators}{PIRADS_pattern}',
            ]:
                matches = re.finditer(pattern, conclusion)
                for lesion_number, match in enumerate(matches):
                    scores_lesion = {'tot': pirads_score_map[match.group("PIRADS")]}
                    scores_lesion['tot_pattern'] = 'conclusion'
                    results += [(lesion_number, scores_lesion)]

                    extraction_type = 5

    # encode extraction type in scores
    for lesion_number, scores_lesion in results:
        scores_lesion['extraction_type'] = extraction_type

    return results


def extract_lesion_sections(
    report: str,
    ignore_conclusion: bool = True,
    remove_measurements: bool = True,
    verbose=1
) -> List[Tuple[int, str]]:
    """
    Try to split the report is separate sections for each lesion. Many reports allow simple splitting
    by their 'lesion number', as indicated below in examples 1. and 2. If the report could not be split
    reliably, then the full report is returned.

    Typical reports have the following structures:
    1.
        Afwijking nr. 1: ter plaatse van de perifere zone rechts laterodorsaal apex prostaat.
        Score T2W: 5,
        Score DCE: +,
        Score DWI: 5, minimale ADC waarde 524.
        Beeld past het best bij intermediair- tot hooggradig prostaatcarcinoom (PIRADS 5) ...

    2.
        Index laesie mark1:\
        Plaats: transitie zone/uitpuilende nodule in de perifere zone rechts apicaal.\
        T2W/DWI/DCE score: 3/5/+.\
        Minimale ADC waarde: 617 (normaal groter dan 950).\
        Risico categorie: Equivocal (PIRADS v2 categorie: 3).\

    3.
        Ter plaatse van de perifere zone links posterieur mid prostaat (...).
        Score T2W: 2,
        Score DCE: +,
        Score DWI: 2, minimale ADC waarde 1145.
        Beeld past het best bij verdenking focale prostatitis d.d. focus laaggradig carcinoom  (PIRADS 2).
        afmeting van dit gebied bedraagt 11 x6Â mm. Geen tekenen van extra prostatische uitbreiding.

    4.
        Diffuus in vrijwel de gehele prostaat zowel perifere zone als transitie zone, rechts meer dan links.
        Score T2W: 5, Score DCE: +, Score DWI: 5, minimale ADC waarde 507.
        Beeld past het best bij hooggradig prostaatcarcinoom (PIRADS 5).
        ...
        Aankleurend focus in de perifere zone paramediaan rechts geheel apicaal met geringe diffusie-restrictie ...
        Afmeting van deze laesie bedraagt circa 8Â mm.
        Beeld is eveneens verdacht voor een lokaal recidief (Score T2: 3, DCE: +, DWI: 4; PIRADS 4).
        Geen tekenen van extra prostatische uitbreiding.

    Returns:
    - a list of lesion id and the report snippet for that lesion:
        [(1, "Lesion nr. 1: ..."),
         (2, "Lesion nr. 2: ..."),
          ...]
    if the report was not split in sections, the lesion id is 0:
        [(0, full report)]
    """

    if ignore_conclusion:
        report = remove_conclusion_from_report(report)

    if remove_measurements:
        report = re.sub(r"\d+ *[xX] *\d+", "[removed]", report)
        report = re.sub(r"\d+ *cc", "[removed]", report)

    # search for sections
    at_least_one_number = r'(?P<num1>\d+)\+?(?P<num2>\d+)?\+?(?P<num3>\d+)?\+?(?P<num4>\d+)?'
    optional_nr_mark = r' *(nr\.?)? *(mark)? *((in)? nummer)? *'
    prefix_pattern = "|".join(('Afwijking', 'Index laesie', 'Markering', 'Regio', 'Laesie', 'Lesion index'))
    pattern = f'({prefix_pattern}):?{optional_nr_mark}{at_least_one_number}'

    # detect lesion sections
    matches = list(re.finditer(pattern, report, re.IGNORECASE))

    # verbose
    if verbose >= 2:
        for match in matches:
            report_snippet = report[slice(max(0, match.span()[0]-40), match.span()[1]+40)]
            report_snippet = report_snippet.replace('\n', r'\\n')
            print(f"section from: {report_snippet}")

    if len(matches):
        # return start of match to start of next match (or end of report)
        report_snippets = []
        for i, match in enumerate(matches):
            if (i+1) < len(matches):
                report_snippet = report[match.span()[0]:matches[i+1].span()[0]]
            else:
                report_snippet = report[match.span()[0]:]

            # add report snippet for lesion number
            # also allow multiple lesions, as in 'Markering nr. 1+2'
            for key in [f'num{i}' for i in (1, 2, 3, 4)]:
                if match.group(key) is not None:
                    lesion_number = int(match.group(key))
                    report_snippets += [(lesion_number, report_snippet)]

        # fix lesion numbering
        for i, res in enumerate(report_snippets):
            if i == 0:
                continue
            if res[0] == report_snippets[i-1][0]:
                # lesion index is the same
                if verbose >= 2:
                    print(f"Changing index lesion numbering for {report}")
                    print("="*50)
                elif verbose == 1:
                    print("Changing index lesion numbering")
                report_snippets[i] = (report_snippets[i][0] + 1, report_snippets[i][1])
        return report_snippets

    # no sections found, return complete report
    return [(0, report)]


def extract_pirads_scores_from_lesion_section(
    report_snippet: str,
    verbose: int = 1
) -> Dict[str, Union[str, int, None]]:
    scores: Dict[str, Union[str, int, None]] = {
        'T2W': None,
        'DWI': None,
        'DCE': None,
        'tot': None,
        'T2W_pattern': None,
        'DWI_pattern': None,
        'DCE_pattern': None,
        'tot_pattern': None,
    }

    # extract T2W score
    for pattern in [
        r'S?core.? T2?[wW]?: *([1-5])',
        r'T2W/DWI/DCE score: *([1-5])/.{1,5}/[+\-xX]',
        r'T2W?: *([1-5])',
    ]:
        match = re.search(pattern, report_snippet)
        if match is not None:
            score = match.group(1)
            scores['T2W'] = score
            scores['T2W_pattern'] = pattern
            break

    # extract DWI score
    for pattern in [
        r'S?core.? DWI: *([1-5])',
        r'T2W/DWI/DCE score: *.{1,5}/X?-?([1-5])/[+\-xX]',
        r'DWI: *([1-5])',
    ]:
        match = re.search(pattern, report_snippet)
        if match is not None:
            score = match.group(1)
            scores['DWI'] = score
            scores['DWI_pattern'] = pattern
            break

    # extract DCE score
    for pattern in [
        r'S?core.? DCE: *\.?([+\-])',
        r'T2W/DWI/DCE score: *.{1,5}/.{1,5}/([+\-])',
        r'DCE: *([+-])',
    ]:
        match = re.search(pattern, report_snippet)
        if match is not None:
            score = match.group(1)
            scores['DCE'] = score
            scores['DCE_pattern'] = pattern
            break

    # extract total score
    for pattern in [
        fr'PI-?>?RADS v2 (categorie|category){allowed_separators}{PIRADS_pattern}',
        fr'PI-?RADS{allowed_separators}{PIRADS_pattern}',
    ]:
        match = re.search(pattern, report_snippet)
        if match is not None:
            score_tot = pirads_score_map[match.group("PIRADS")]
            scores['tot'] = score_tot
            scores['tot_pattern'] = pattern
            if verbose >= 2:
                print(f"Found {score} with {pattern} in: {report_snippet}")
            break

    if scores['tot'] is None and verbose >= 2:
        print("tot score not found for: ", report_snippet)
        print("="*50)

    return scores


def extract_all_scores_from_full_report(
    report: str,
    subject_id: Optional[str] = None,
    strict: bool = False,
    aggressive: bool = False,
    verbose: int = 1
) -> List[Tuple[int, Dict[str, Union[str, int, None]]]]:
    """Extract full scores from snippets like:
    Score T2W: 5, Score DCE: +, Score DWI: 5, minimale ADC waarde 507.
    Beeld past het best bij hooggradig prostaatcarcinoom (PIRADS 5).
    ...
    Aankleurend focus in de perifere zone paramediaan rechts geheel apicaal met geringe diffusie-restrictie ...
    Afmeting van deze laesie bedraagt circa 8Â mm.
    Beeld is eveneens verdacht voor een lokaal recidief (Score T2: 3, DCE: +, DWI: 4; PIRADS 4).
    """
    all_scores: List[Tuple[int, Dict[str, Union[str, int, None]]]] = []
    num = 0
    T2W_score = r'(?P<T2W>[0-5xX-])'
    DWI_score = r'(?P<DWI>[0-5xX\-])'
    DCE_score = r'(?P<DCE>[+\-xX\/]+)'

    pattern_list = [
        # composite scores
        fr'T2W?/DWI/DCE *(score)?:? *{T2W_score}/{DWI_score}/{DCE_score}',
        fr'T2W?:? *{T2W_score} *[,;]? *(Score)? *DCE:? *{DCE_score} *[,;]? *(Score)? *DWI:? *{DWI_score}',
        fr'T2W?:? *{T2W_score} *[,;]? *DWI:? *{DWI_score} *(\(?ADC waarden? *(tot)? *\d+\)?)? *[,;]? *DCE: {DCE_score}',

        # match Score T2W: 5,\n Score DCE: +, \nScore DWI: 5, minimale ADC waarde 648:
        fr'(Score)? *T2W:? *{T2W_score} *[,;]?\\\\? *\n(Score)? *DCE:? *{DCE_score} *[,;]?\\\\? *\n(Score)? *DWI:? *{DWI_score}',
        fr'(Score)? *T2W:? *{T2W_score}{allowed_separators}(Score)? *DCE:? *{DCE_score}{allowed_separators}(Score)? *DWI:? *{DWI_score}',

        # edge cases:
        fr'T2W?:? *{T2W_score} *[,;]? *(Score)? *ADC:? *{DCE_score} *[,;]? *(Score)? *DWI:? *{DWI_score}',
        fr'T2W?:? *{T2W_score} *[,;]? *(Score)? *DWI:? *{DWI_score} *[,;]? *ADC waarden circa (\d+)[,;]? *(Score)? *DCE: *{DCE_score}',
        fr'T2W?:? *(score)?:? *{T2W_score}[,;]? *DWI:? *{DWI_score}[,;]? *DCE:? *{DCE_score}',

        fr'T2W?:? *{T2W_score},?\n?DWI:? *{DWI_score} \(ADC-waarde: \d+\)?,?\n?DCE:? *{DCE_score}',
        fr'T2W?:? *{T2W_score},? *\n? *(Score)? ?DWI:? *{DWI_score}, de minimale ADC waarde is \d+ *,?.? *\n? *(Score)? DCE:? *{DCE_score}',
        fr'T2W?:? *{T2W_score},? *\n? *(Score)? ?\n?DWI:? *{DWI_score}, de minimale ADC waarde is normaal.? *\n? *,? *\n?(Score)? DCE:? *{DCE_score}',
    ]

    if not strict:
        pattern_list += [
            r'T2W?:? *%s.{0,10}\n?(Score)? *DWI:? *%s.{0,100}\n?(Score)? *DCE:? *%s' % (T2W_score, DWI_score, DCE_score),
        ]

    for pattern in pattern_list:
        while True:
            match = re.search(pattern, report, re.IGNORECASE)

            if match is None:
                break

        # matches = re.finditer(pattern, report, re.IGNORECASE)
        # for match in matches:
            scores: Dict[str, Union[str, int, None]] = {
                'T2W': None,
                'DWI': None,
                'DCE': None,
                'tot': None,
                'T2W_pattern': None,
                'DWI_pattern': None,
                'DCE_pattern': None,
                'tot_pattern': None,
            }

            # select matches
            T2W, DWI, DCE = match.group('T2W'), match.group('DWI'), match.group('DCE')

            # decide on DCE
            if DCE == '+/-' or DCE == '-/+':
                DCE = ''
            elif DCE == '++':
                DCE = '+'
            elif DCE != '-' and DCE != '+':
                print("ood dce: ", DCE)

            # store scores
            scores['T2W'] = T2W
            scores['DWI'] = DWI
            scores['DCE'] = DCE

            # store patterns
            scores['T2W_pattern'] = scores['DWI_pattern'] = scores['DCE_pattern'] = pattern

            # store
            num += 1
            all_scores += [(num, scores)]

            # remove snippet from report to prevent double matches
            report = report.replace(match.group(0), '', 1)  # 1: only replace first occurrence

    # extract total scores
    num_tot = 0
    for pattern in [
        fr'PI-?>?RADS v2 (categorie|category){allowed_separators}{PIRADS_pattern}',
        fr'PI-?>?RADS{allowed_separators}{PIRADS_pattern}',
    ]:
        # cross fingers the order of matches is the same
        matches = re.finditer(pattern, report)

        for match in matches:
            if num_tot < len(all_scores):
                _, scores = all_scores[num_tot]
                scores['tot'] = pirads_score_map[match.group("PIRADS")]
                scores['tot_pattern'] = pattern
                num_tot += 1
            elif verbose >= 2:
                print("Ignoring PI-RADS match: ", match.group(0))

    if num_tot != len(all_scores) and not aggressive:
        # matching failed, return None
        raise Exception(f"Matching of individual scores with resulting PI-RADS scores failed for {subject_id}, and aggressive={aggressive}" +
                        f" (found individual scores for {len(all_scores)} lesions and total PI-RADS for {num_tot})")

    if verbose >= 2:
        print("All scores from full report:", all_scores)

    return all_scores


def remove_addendum_from_report(report: str) -> str:
    """Renove the addendum from a report.

    The addendum is expected to be in the following format:
    ---------------------------------------------Addendum start---------------------------------------------
    [Addendum text]
    ---------------------------------------------Addendum einde---------------------------------------------
    [Rest of report]
    """
    while True:
        match_start = re.search('-*Addendum start-*', report)
        match_stop = re.search('-*Addendum einde-*', report)
        if match_start is not None and match_stop is not None:
            # remove addendum from report by selecting the text before and after the addendum
            report = report[:match_start.span()[0]] + report[match_stop.span()[1]:]
        else:
            break
    return report


def remove_after(pattern: str, report: str, ignore_case: bool = True) -> str:
    match = re.search(pattern, report, (re.IGNORECASE if ignore_case else 0))
    if match is not None:
        # remove part from report by selecting the text before
        report = report[:match.span()[0]]

    return report


def remove_conclusion_from_report(report: str) -> str:
    """Renove the conclusion/radiologist's impression from a report.

    The conclusion/radiologist's impression is expected to be in the following format:
    [Report body]
    Impressie:
    [conclusion/radiologist's impression]
    """
    return remove_after("Impressie:", remove_after("Conclusie:", report))


allowed_separators = r"[,;:\n ]*"
PIRADS_pattern = "(?P<PIRADS>[1-5]|een|twee|drie|vier|vijf)"
pirads_score_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                    'een': 1, 'twee': 2, 'drie': 3, 'vier': 4, 'vijf': 5}
