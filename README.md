# Schedule E PDF Extractor

This tool reads Schedule E Part II PDF forms and creates a table of partnership
and S-corporation income or loss details. It works with both searchable PDFs
and scanned PDFs.

## What It Creates

For each PDF, the extractor creates a CSV file with these columns:

- Name
- P/S
- EIN
- Passive loss
- Passive income
- Non-passive loss
- Section 179 deduction
- Non-passive income

It also creates a processed PDF in `work/`. This file is used during OCR and
can be deleted after you have checked the results.

## Project Folders

```text
input/     Put source PDFs here. Subfolders are supported.
output/    Generated CSV files are written here.
work/      Temporary OCR files are written here.
tests/     Test results and validation metrics.
src/       The Schedule E extractor code.
```

When source PDFs are inside test folders, their subfolder names are kept in
`output/` and `work/`. This prevents files with the same name from overwriting
each other.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For scanned PDFs, install Tesseract OCR and make sure its English language data
is available to PyMuPDF.

## Run The Extractor

From the project folder, run:

```bash
PYTHONPATH=src python -m schedule_e_pdf_extractor
```

The program searches for every PDF below `input/` and writes matching CSV files
below `output/`. Excel files can be created from validated CSV results during a
separate export step.

## Understanding The Result

The extractor writes a CSV only when the data passes its reliability check.
If a PDF does not create an output file, review the console message and the
processed PDF in `work/`. This protects you from receiving a table with shifted
names, EINs, or values.

## Code Layout

- `part1.py` prepares PDFs, runs OCR, detects lines, and identifies headers.
- `part2.py` builds table columns and rows, cleans values, scores confidence,
  and processes folders of PDFs.
- `__main__.py` is the command-line entry point.
