# Extraction Validation Report

## Result

No CSV output was generated. All 10 supplied PDFs were attempted twice, and `output/` contains zero CSV files. Therefore no extracted values can be verified as correct.

## Input Results

| Input group | Result | Cause |
| --- | --- | --- |
| Tests 01-06 and 08-10 | Failed before table extraction | OCR-produced PDF contains zero characters. The empty line table reaches `header_delimiter`, which calls `groupby("page")` even though the table has no `page` column. |
| Test 07 (`native_duplicated_chars`) | Failed during row reconstruction | `join_to_char_df_uncollapsed` expects `comb_line_num`, but that column was not created. |

## Verified Evidence

- The nine scan/encoded cases have zero characters after the OCR preprocessing step.
- Test 07 preserves 576 processed characters but still fails before CSV output.
- The failures reproduce when processing the batch and when processing representative failing inputs directly with full stack traces.

## Required Fixes Before Validation Can Continue

1. Install a local OCR engine. The revised implementation now uses PyMuPDF's searchable-PDF OCR API, which requires Tesseract and its language data.
2. Rerun the test set after OCR is available and compare each CSV to the source PDF.

## Implemented Corrections

- OCR preprocessing now creates a searchable PDF instead of invoking markdown generation that leaves the PDF unchanged.
- Empty character and line tables now raise descriptive `ValueError` exceptions before header detection.
- Logical row offsets no longer use pandas `groupby.apply`, which dropped `comb_line_num` under pandas 3.
- Word spacing uses logical rows before calculating gaps and preserves dollar signs.
- Header selection and table reconstruction include page identity, preventing multipage rows from being merged.
- CSV writing is blocked below a 75% confidence score, preventing known-bad output from being saved.
