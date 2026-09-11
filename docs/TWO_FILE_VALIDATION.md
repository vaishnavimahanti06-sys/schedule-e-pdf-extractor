# Two-File Validation

## Generated Outputs

- `schedule_e_test_01_clean_baseline.csv`: 5 rows, 100.0% confidence.
- `schedule_e_test_02_multiline_wrap.csv`: 6 rows, 95.8% confidence.

## Verification

The names, P/S classifications, EIN/APPLIED FOR values, and each of the five numeric fields were compared to the source PDF pages.

| Input | Rows | Result |
| --- | ---: | --- |
| `schedule_e_test_01_clean_baseline.pdf` | 5 | All values match. Extracted totals match the printed PDF totals: 1,600 / 4,450 / 300 / 1,250 / 7,150. |
| `schedule_e_test_02_multiline_wrap.pdf` | 6 | All visible row values match. The extracted row totals are 4,150 / 4,450 / 200 / 800 / 9,950. The source PDF prints 10,450 for its final Non-Passive Income total, which is 500 higher than its own visible rows. |

## Source Total Discrepancy

The final wrapped entity has Non-Passive Income of 500. Summing the six displayed rows gives 9,950, while the PDF prints 10,450. The CSV preserves the row-level source values and therefore totals 9,950. This is a source-document inconsistency, not an extraction error.
