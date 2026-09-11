# Full Input Validation

## Outcome

All 10 input PDFs were processed. Six produced reviewed CSV outputs; four were correctly withheld because their extraction quality did not meet the 75% confidence threshold.

## Approved Outputs

| Input | Output | Rows | Confidence | Validation |
| --- | --- | ---: | ---: | --- |
| `01_clean_baseline` | `01_clean_baseline.csv` | 5 | 100.0% | Rows and printed totals match. |
| `02_multiline_wrap` | `02_multiline_wrap.csv` | 6 | 95.8% | Wrapped entity names merged correctly. See source-total note below. |
| `04_wide_column_spacing` | `04_wide_column_spacing.csv` | 4 | 100.0% | All rows have valid P/S, EIN, and value columns. |
| `05_no_header_delimiter` | `05_no_header_delimiter.csv` | 3 | 100.0% | Dynamic geometry correctly reconstructed columns. |
| `09_multipage_totals_statement` | `09_multipage_totals_statement.csv` | 6 | 100.0% | Page footer leakage was removed; page-one and page-two entities remain separate. |
| `10_ssn_box_masked_ein` | `10_ssn_box_masked_ein.csv` | 3 | 83.3% | Row values are extracted; two EINs remain intentionally masked with `*` characters. |

## Withheld Inputs

| Input | Reason |
| --- | --- |
| `03_rotated_sideways_scan` | OCR text remains rotated/garbled; 25.0% confidence. |
| `06_native_encoded_text` | OCR cannot reliably reconstruct the encoded source; 12.5% confidence. |
| `07_native_duplicated_chars` | No reliable numeric anchors after deduplication and multiline reconstruction. |
| `08_degraded_noisy_scan` | OCR is too degraded to recover names and identifiers reliably; 50.0% confidence. |

## Source-Document Note

For test 02, visible rows sum to 9,950 in Non-Passive Income, while the PDF prints 10,450. The CSV retains the visible row values and therefore totals 9,950. This is a source-document inconsistency, not a CSV extraction error.
