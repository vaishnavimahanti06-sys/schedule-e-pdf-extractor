"""PDF preparation, OCR, line detection, and row classification helpers."""

import os
import re
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pdfplumber
import pymupdf
import pymupdf4llm
import cv2
import pytesseract

from pypdf import PdfReader, PdfWriter



warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:.2f}".format

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

type_attributes = {}



var1 = ["INCOMEOR"]

var2 = [
    "LOSS",
    "(LOSS)"
]

var3 = [
    "FROMPARTNERSHIPSANDS"
]

var4 = [
    "CORPORATIONS",
    "CORPS"
]

vars = [
    var1,
    var2,
    var3,
    var4
]

line_variations = [
    "".join(x)
    for x in product(*vars)
]


EIN_PAT = (
    r"[\d*]{2}-[\d*]{7}"
)


EIN_OR_APPLIEDFOR_PAT = (
    rf"(?:{EIN_PAT}|APPLIEDFOR)"
)


VALUE_PAT = (
    r"\$?"
    r"(?:"
    r"NONE\.?"
    r"|"
    r"(?:\d{1,3}(?:,\d{3})*|\d+)\.?"
    r")"
)


PS_LEAD_PAT = (
    r"^[SP](?:[X*]+)?"
    r"(?=(?:\$|NONE|\d|,|\.|$))"
)


VALUE_LABELS = [
    "PASSIVE LOSS",
    "PASSIVE INCOME",
    "NON-PASSIVE LOSS",
    "SECTION 179 DEDUCTION",
    "NON-PASSIVE INCOME",
]



def native_text_quality(pdf_path):
    """
    Check whether the PDF's native text layer is reliable.

    Returns:
        bad_text: bool
        diagnostics: dict
    """

    doc = pymupdf.open(pdf_path)

    total_chars = 0
    useful_chars = 0
    bad_chars = 0
    private_chars = 0
    replacement_chars = 0

    page_texts = []

    for page in doc:

        text = page.get_text("text") or ""

        page_texts.append(text)

        total_chars += len(text)

        for ch in text:

            code = ord(ch)

            if ch.isalnum():
                useful_chars += 1

            if ch == "\ufffd":
                replacement_chars += 1
                bad_chars += 1

            elif 0xE000 <= code <= 0xF8FF:
                private_chars += 1
                bad_chars += 1

            elif code < 32 and not ch.isspace():
                bad_chars += 1

    doc.close()

    complete_text = "\n".join(page_texts)

    normalized = re.sub(
        r"\s+",
        "",
        complete_text.upper()
    )

    bad_ratio = (
        bad_chars
        / max(total_chars, 1)
    )

    ein_matches = re.findall(
        r"\b\d{2}-\d{7}\b",
        complete_text
    )

    schedule_e_found = (
        "SCHEDULEE"
        in normalized
    )

    part_ii_found = (
        "PARTII"
        in normalized
        or
        "PARTNERSHIPSANDSCORPORATIONS"
        in normalized
    )

    reasons = []

    if total_chars < 50:
        reasons.append(
            "Very little native text"
        )

    if useful_chars < 30:
        reasons.append(
            "Very little readable text"
        )

    if bad_ratio > 0.03:
        reasons.append(
            "High unreadable-character ratio"
        )

    if private_chars > 5:
        reasons.append(
            "Private-use encoded characters detected"
        )

    if replacement_chars > 5:
        reasons.append(
            "Unicode replacement characters detected"
        )

    # Some PDFs expose an encoded font as apparently readable alphanumeric
    # text. Treat a layer with neither form anchors nor EINs as unreliable,
    # so the OCR path can recover the actual table content.
    if (
        len(ein_matches) == 0
        and not schedule_e_found
        and not part_ii_found
    ):
        reasons.append(
            "Native text lacks Schedule E and EIN anchors"
        )

    if (
        schedule_e_found
        and
        part_ii_found
        and
        len(ein_matches) == 0
        and
        useful_chars < 250
    ):
        reasons.append(
            "Schedule E detected but EIN/data extraction is weak"
        )

    bad_text = (
        len(reasons) > 0
    )

    diagnostics = {
        "total_chars": total_chars,
        "useful_chars": useful_chars,
        "bad_chars": bad_chars,
        "bad_ratio": round(
            bad_ratio,
            4
        ),
        "private_chars": private_chars,
        "replacement_chars": replacement_chars,
        "ein_count": len(ein_matches),
        "schedule_e_found": schedule_e_found,
        "part_ii_found": part_ii_found,
        "ocr_required": bad_text,
        "reasons": reasons,
    }

    return (
        bad_text,
        diagnostics
    )



def rasterize_pdf(
    input_pdf,
    output_pdf,
    dpi=300
):
    """
    Convert PDF pages into image-only pages.

    This removes dependency on a corrupted embedded
    text layer before OCR is applied.
    """

    source = pymupdf.open(
        input_pdf
    )

    output = pymupdf.open()

    zoom = dpi / 72

    matrix = pymupdf.Matrix(
        zoom,
        zoom
    )

    for source_page in source:

        pix = source_page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        page_width = (
            pix.width
            * 72
            / dpi
        )

        page_height = (
            pix.height
            * 72
            / dpi
        )

        new_page = output.new_page(
            width=page_width,
            height=page_height
        )

        rect = pymupdf.Rect(
            0,
            0,
            page_width,
            page_height
        )

        new_page.insert_image(
            rect,
            pixmap=pix
        )

    output.save(
        str(output_pdf),
        garbage=4,
        deflate=True
    )

    source.close()
    output.close()

    return str(
        output_pdf
    )


def create_searchable_ocr_pdf(
    input_pdf,
    output_pdf,
    language="eng",
    dpi=300
):
    """Create a PDF whose page images have an embedded OCR text layer."""
    source = pymupdf.open(input_pdf)
    output = pymupdf.open()

    try:
        for source_page in source:
            page = output.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height
            )
            page.show_pdf_page(
                page.rect,
                source,
                source_page.number
            )
            text_page = source_page.get_textpage_ocr(
                language=language,
                dpi=dpi,
                full=True
            )
            ocr_data = source_page.get_text(
                "rawdict",
                textpage=text_page
            )
            for block in ocr_data["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        for char in span["chars"]:
                            x0, y0, x1, y1 = char["bbox"]
                            page.insert_text(
                                (x0, y1),
                                char["c"],
                                fontsize=max(y1 - y0, 1),
                                fontname="helv",
                                render_mode=3
                            )

        output.save(
            str(output_pdf),
            garbage=4,
            deflate=True
        )
    finally:
        source.close()
        output.close()

    return str(output_pdf)


def detect_scan_orientation(image):
    """Score each cardinal orientation with a small Schedule E OCR pass."""
    image = image.copy()
    max_dimension = 1400
    scale = min(1.0, max_dimension / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    rotations = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    terms = (
        "schedule e", "form 1040", "part ii", "income",
        "partnerships", "corporations", "ein", "total", "name",
    )
    scores = {}
    for angle, rotated in rotations.items():
        try:
            text = pytesseract.image_to_string(
                rotated,
                config="--psm 6",
                timeout=15,
            )
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", " ", text.lower())
        score = sum(term in normalized for term in terms)
        if re.search(r"\b\d{2}[-–]\d{7}\b", normalized):
            score += 2
        if "$" in text:
            score += 1
        scores[angle] = score

    return max(scores, key=scores.get), scores


def enhance_scan_for_ocr(image):
    """Apply local contrast enhancement without changing page geometry."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(image)


def remove_table_rules_for_ocr(image):
    """Remove long horizontal and vertical table rules while retaining text."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.threshold(
        image,
        200,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]
    horizontal = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)),
    )
    vertical = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)),
    )
    cleaned = image.copy()
    cleaned[(horizontal | vertical) > 0] = 255
    return cleaned


def score_ocr_candidate(image):
    """Score a full-resolution OCR candidate without trusting one signal."""
    text = pytesseract.image_to_string(
        image,
        config="--psm 6",
        timeout=30,
    )
    normalized = re.sub(r"\s+", " ", text.lower())
    terms = (
        "schedule e", "form 1040", "part ii", "income",
        "partnerships", "corporations", "ein", "total", "name",
    )
    score = sum(term in normalized for term in terms)
    # Accept OCR spaces around the EIN dash.
    score += 5 * len(re.findall(r"\b\d{2}\s*[-–]\s*\d{7}\b", text))
    score += text.count("$")
    return score, text


def rasterize_pdf_for_ocr(input_pdf, output_pdf, dpi=300):
    """Correct scan orientation and produce the strongest searchable OCR PDF."""
    source = pymupdf.open(input_pdf)
    output = pymupdf.open()
    diagnostics = []

    try:
        for page_number, page in enumerate(source, start=1):
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                colorspace=pymupdf.csGRAY,
                alpha=False,
            )
            image = np.frombuffer(
                pixmap.samples,
                dtype=np.uint8,
            ).reshape(pixmap.height, pixmap.width)

            angle, orientation_scores = detect_scan_orientation(image)
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            candidates = {
                "raw_grayscale": image,
                "clahe": enhance_scan_for_ocr(image),
                "table_rules_removed": remove_table_rules_for_ocr(image),
            }
            candidate_scores = {
                name: score_ocr_candidate(candidate)[0]
                for name, candidate in candidates.items()
            }
            # Prefer raw grayscale when OCR scores tie.
            selected_treatment = max(
                candidate_scores,
                key=lambda name: (
                    candidate_scores[name],
                    name == "raw_grayscale",
                ),
            )
            selected = candidates[selected_treatment]
            raw_score = candidate_scores["raw_grayscale"]
            clahe_score = candidate_scores["clahe"]

            height, width = selected.shape
            rgb = cv2.cvtColor(selected, cv2.COLOR_GRAY2RGB)
            ocr_pixmap = pymupdf.Pixmap(
                pymupdf.csRGB,
                width,
                height,
                rgb.tobytes(),
                False,
            )
            page_pdf = pymupdf.open(
                "pdf",
                ocr_pixmap.pdfocr_tobytes(language="eng"),
            )
            output.insert_pdf(page_pdf)
            page_pdf.close()
            diagnostics.append({
                "page": page_number,
                "rotation": angle,
                "orientation_scores": orientation_scores,
                "raw_score": raw_score,
                "clahe_score": clahe_score,
                "table_rules_removed_score": candidate_scores[
                    "table_rules_removed"
                ],
                "selected_treatment": selected_treatment,
                "width": selected.shape[1],
                "height": selected.shape[0],
            })

        output.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        source.close()
        output.close()

    return str(output_pdf), diagnostics


def require_extractable_characters(char_df, context):
    """Reject PDFs that cannot support coordinate-based extraction."""
    required_columns = {
        "text",
        "x0",
        "x1",
        "top",
        "bottom",
        "page"
    }
    missing_columns = required_columns - set(char_df.columns)

    if missing_columns:
        raise ValueError(
            f"{context}: missing character columns "
            f"{sorted(missing_columns)}."
        )

    if char_df.empty:
        raise ValueError(
            f"{context}: no extractable characters were found. "
            "The PDF needs a usable OCR text layer."
        )



def prepare_pdf_with_pymupdf4llm(
    input_pdf,
    working_folder=None,
    force_ocr=None,
    ocr_language="eng",
    ocr_dpi=300
):
    """
    GOOD PDF:
        keep native text and use hybrid OCR.

    BAD PDF:
        rasterize first, then force OCR.

    Returns:
        processed_pdf
        diagnostics
    """

    print(
        "---- CHECKING NATIVE PDF TEXT QUALITY ----"
    )

    (
        detected_bad_text,
        diagnostics
    ) = native_text_quality(
        input_pdf
    )

    print(
        diagnostics
    )

    if force_ocr is None:
        force_ocr = detected_bad_text

    source_path = Path(
        input_pdf
    )

    if working_folder is None:

        work_dir = (
            source_path.parent
            / "_schedule_e_work"
        )

    else:

        work_dir = Path(
            working_folder
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_pdf = (
        work_dir
        /
        f"{source_path.stem}_processed.pdf"
    )

    if force_ocr:
        print("---- OCR: ORIENTATION, CLAHE, SEARCHABLE PDF ----")
        processed_pdf, ocr_pages = rasterize_pdf_for_ocr(
            input_pdf,
            processed_pdf,
            dpi=ocr_dpi
        )
        diagnostics["ocr_pages"] = ocr_pages
        with pdfplumber.open(processed_pdf) as pdf:
            diagnostics["ocr_chars"] = sum(len(page.chars) for page in pdf.pages)
            diagnostics["ocr_text_length"] = sum(
                len(page.extract_text() or "")
                for page in pdf.pages
            )
        if diagnostics["ocr_chars"] == 0:
            raise ValueError("OCR preprocessing produced zero pdfplumber characters.")
    else:
        # Keep native coordinates for pdfplumber.
        doc = pymupdf.open(input_pdf)
        try:
            doc.save(
                str(processed_pdf),
                garbage=4,
                deflate=True
            )
        finally:
            doc.close()

    diagnostics[
        "force_ocr_used"
    ] = bool(
        force_ocr
    )

    diagnostics[
        "processed_pdf"
    ] = str(
        processed_pdf
    )

    return (
        str(processed_pdf),
        diagnostics
    )



def to_rotate(in_page):
    """Detect whether a PDF page is landscape from character dimensions."""

    char_df = pd.DataFrame(
        in_page.chars
    )

    if char_df.empty:

        type_attributes[
            "landscape"
        ] = False

        return False

    char_width = (
        char_df[
            "width"
        ].mean()
    )

    char_height = (
        char_df[
            "height"
        ].mean()
    )

    if char_width > char_height:

        type_attributes[
            "landscape"
        ] = True

        return True

    type_attributes[
        "landscape"
    ] = False

    return False


def rotate(
    in_file,
    rotation_degrees=90
):
    """Write a rotated portrait copy of the supplied PDF."""

    reader = PdfReader(
        in_file
    )

    writer = PdfWriter()

    cleaned = (
        str(in_file)
        .rsplit(
            ".pdf",
            1
        )[0]
    )

    out_file = (
        cleaned
        +
        " - Portrait.pdf"
    )

    for page in reader.pages:

        rotated = page.rotate(
            rotation_degrees
        )

        writer.add_page(
            rotated
        )

    with open(
        out_file,
        "wb"
    ) as f:

        writer.write(
            f
        )

    return out_file



def to_check_decode(
    in_page
):
    """Read the upper-right identifier area used to detect encoded text."""

    x = in_page.width
    y = in_page.height

    bbox = (
        x * 0.75,
        0,
        x,
        y * 0.05
    )

    crop_page = (
        in_page.crop(
            bbox
        )
    )

    text = (
        crop_page.extract_text()
    )

    if text:

        ssn = (
            text
            .strip(
                "\uf040"
            )[0:11]
        )

        type_attributes[
            "ssn_box"
        ] = True

        return ssn

    type_attributes[
        "ssn_box"
    ] = False

    return None


def decode(s):
    """Decode the document's custom character mapping in text or a DataFrame."""

    az_map = {
        chr(c):
        chr(
            ord("!")
            +
            (
                c
                -
                ord("A")
            )
        )
        for c in range(
            ord("A"),
            ord("Z") + 1
        )
    }

    private_use_map = {
        chr(
            0xF020 + c
        ):
        chr(c)
        for c in range(
            0x20,
            0x7F
        )
    }

    special = {
        "@": " ",
        "\x9e": "}"
    }

    decode_map = {
        **az_map,
        **private_use_map,
        **special
    }

    def _decode_one(x):
        """Decode and normalize one encoded text value."""

        decoded = "".join(
            decode_map.get(
                ch,
                ch
            )
            for ch in str(x)
        )

        return (
            decoded.capitalize()
        )

    if isinstance(
        s,
        str
    ):

        return _decode_one(
            s
        )

    elif isinstance(
        s,
        pd.DataFrame
    ):

        s = s.copy()

        s[
            "text"
        ] = (
            s[
                "text"
            ]
            .apply(
                _decode_one
            )
        )

        return s

    return (
        "Incorrect format"
    )


def to_decode(s):
    """Determine whether an identifier needs decoding and return its usable value."""

    if not s:

        type_attributes[
            "encoded"
        ] = False

        return (
            False,
            None
        )

    original_match = bool(
        re.fullmatch(
            r"[^-]{3}-[^-]{2}-[^-]{4}",
            s
        )
    )

    decoded = decode(
        s
    )

    decoded_match = bool(
        re.fullmatch(
            r"[^-]{3}-[^-]{2}-[^-]{4}",
            decoded
        )
    )

    if original_match:

        type_attributes[
            "encoded"
        ] = False

        return (
            False,
            s
        )

    elif decoded_match:

        type_attributes[
            "encoded"
        ] = True

        return (
            True,
            decoded
        )

    type_attributes[
        "encoded"
    ] = False

    return (
        False,
        None
    )



def to_dedup(
    in_page
):
    """Detect duplicate page characters and return a deduplicated character table."""

    char_df = pd.DataFrame(
        [
            (
                c["text"],
                c["x0"],
                c["bottom"],
                c["x1"],
                c["top"],
                c["page_number"] - 1
            )
            for c in in_page.chars
        ],
        columns=[
            "text",
            "x0",
            "bottom",
            "x1",
            "top",
            "page"
        ]
    )

    deduped_char_df = (
        char_df
        .drop_duplicates()
    )

    has_duplicates = (
        not deduped_char_df.equals(
            char_df
        )
    )

    type_attributes[
        "duplicated"
    ] = has_duplicates

    return (
        has_duplicates,
        deduped_char_df
    )


def dedup(
    in_page
):
    """Remove repeated characters from a pdfplumber page for extraction."""

    char_df = pd.DataFrame(
        in_page.chars
    )

    deduped_char_df = (
        char_df
        .drop_duplicates(
            subset=[
                "text",
                "x0",
                "bottom"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        deduped_char_df
        .to_dict(
            orient="records"
        )
    )



def lines(
    in_char_df
):
    """Group positioned characters into ordered text lines and line metadata."""

    df = (
        in_char_df.copy()
    )

    is_space = (
        df[
            "text"
        ]
        .astype(str)
        .str.fullmatch(
            r"\s+"
        )
        .fillna(False)
    )

    df = (
        df.loc[
            ~is_space
        ]
        .copy()
    )

    df = (
        df
        .sort_values(
            [
                "page",
                "top",
                "x0"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "x_delta"
    ] = (
        df[
            "x0"
        ].diff()
    )

    df[
        "y_delta"
    ] = (
        df[
            "top"
        ].diff()
    )

    page_changed = (
        df[
            "page"
        ]
        !=
        df[
            "page"
        ].shift()
    )

    y_changed = (
        df[
            "y_delta"
        ]
        .abs()
        .gt(
            0.75
        )
    )

    df[
        "new_line"
    ] = (
        page_changed
        |
        y_changed
    )

    df[
        "line_num"
    ] = (
        df[
            "new_line"
        ]
        .cumsum()
    )

    tmp = (
        df
        .sort_values(
            [
                "page",
                "line_num",
                "x0"
            ]
        )
    )

    line_df = (
        tmp
        .groupby(
            [
                "page",
                "line_num"
            ],
            as_index=False
        )
        .agg(
            x0=(
                "x0",
                "min"
            ),
            x1=(
                "x1",
                "max"
            ),
            text=(
                "text",
                "".join
            )
        )
    )

    line_df[
        "text"
    ] = (
        line_df[
            "text"
        ]
        .str.lstrip()
    )

    return (
        line_df,
        df
    )



def to_collapse(
    in_line_df,
    min_all_info_rows=3
):
    """Decide whether data entities span multiple lines and need collapsing."""

    df = (
        in_line_df.copy()
    )

    def has_all_info(text):
        """Check whether one line contains identity and value information."""

        s = re.sub(
            r"\s+",
            "",
            str(text)
        )

        pattern = re.compile(
            r".*[SP][X*]*"
            r"(?:"
            r"[\d*]{2}-[\d*]{7}"
            r"|"
            r"X{6}\d{4}"
            r")"
            r"(?:"
            r"NONE\.?"
            r"|"
            r"\$?(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")\.?"
            r")"
        )

        return bool(
            pattern.search(
                s
            )
        )

    df[
        "all_info"
    ] = (
        df[
            "text"
        ]
        .apply(
            has_all_info
        )
    )

    normalized = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    identity_line = normalized.str.match(
        rf"^[SP][X*]*(?:{EIN_OR_APPLIEDFOR_PAT})"
    )
    prior_text = normalized.groupby(df["page"], sort=False).shift(1).fillna("")
    preceding_name_line = (
        prior_text.str.contains(r"[A-Z]", regex=True)
        & ~prior_text.str.contains(EIN_OR_APPLIEDFOR_PAT, regex=True)
    )
    two_line_entities = int((identity_line & preceding_name_line).sum()) >= 2

    type_attributes["two_line_entities"] = two_line_entities

    collapse_bool = bool(
        df["all_info"].sum() < min_all_info_rows
        or two_line_entities
    )

    type_attributes[
        "data_spans_rows"
    ] = collapse_bool

    return (
        df,
        collapse_bool
    )



def header_delimiter(
    in_line_df
):
    """Locate the table-header delimiter used to separate form text from rows."""

    df = (
        in_line_df.copy()
    )

    df[
        "table_top"
    ] = (
        df[
            "text"
        ]
        .apply(
            lambda x:
            any(
                a in str(x)
                for a in line_variations
            )
        )
    )

    cutoff = (
        df[
            df[
                "table_top"
            ]
        ]
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .max()
    )

    df = (
        df[
            df[
                "line_num"
            ]
            >
            df[
                "page"
            ]
            .map(
                cutoff
            )
            .fillna(-1)
        ]
        .reset_index(
            drop=True
        )
    )

    df[
        "footer"
    ] = (
        df[
            "text"
        ]
        .apply(
            lambda x:
            "STATEMENT"
            in str(x)
        )
        &
        (
            df
            .groupby(
                "page"
            )[
                "line_num"
            ]
            .transform(
                "max"
            )
            -
            df[
                "line_num"
            ]
            <
            5
        )
    )

    cutoff = (
        df[
            df[
                "footer"
            ]
        ]
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .min()
    )

    df = (
        df[
            df[
                "line_num"
            ]
            <
            df[
                "page"
            ]
            .map(
                cutoff
            )
            .fillna(
                float("inf")
            )
        ]
        .reset_index(
            drop=True
        )
    )

    df[
        "header_delimiter"
    ] = False

    df[
        "delimiter_character"
    ] = pd.NA

    df[
        "final_header_line"
    ] = False

    if df.empty:

        type_attributes[
            "delimiter"
        ] = False

        type_attributes[
            "delimiter_char"
        ] = None

        return (
            df,
            False,
            None
        )

    df = (
        df
        .sort_values(
            [
                "page",
                "line_num"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df[
        "row_in_page"
    ] = (
        df
        .groupby(
            "page"
        )
        .cumcount()
        + 1
    )

    df[
        "rows_on_page"
    ] = (
        df
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .transform(
            "size"
        )
    )

    check_df = (
        df[
            df[
                "row_in_page"
            ]
            <=
            (
                df[
                    "rows_on_page"
                ]
                / 2
            )
        ]
        .copy()
    )

    check_df[
        "text_nospace"
    ] = (
        check_df[
            "text"
        ]
        .astype(str)
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    mask = (
        check_df[
            "text_nospace"
        ]
        .str.len()
        .ge(4)
        &
        check_df[
            "text_nospace"
        ]
        .apply(
            lambda x:
            len(
                set(x)
            )
            ==
            1
        )
    )

    delim_rows = (
        check_df
        .loc[
            mask,
            [
                "page",
                "line_num",
                "text_nospace"
            ]
        ]
        .copy()
    )

    if delim_rows.empty:

        type_attributes[
            "delimiter"
        ] = False

        type_attributes[
            "delimiter_char"
        ] = None

        return (
            df,
            False,
            None
        )

    delim_rows[
        "delimiter_character"
    ] = (
        delim_rows[
            "text_nospace"
        ]
        .str[0]
    )

    delim_map = (
        delim_rows
        .set_index(
            [
                "page",
                "line_num"
            ]
        )[
            "delimiter_character"
        ]
    )

    keys = list(
        zip(
            df[
                "page"
            ],
            df[
                "line_num"
            ]
        )
    )

    df[
        "delimiter_character"
    ] = (
        pd.Series(
            keys,
            index=df.index
        )
        .map(
            delim_map
        )
    )

    df[
        "header_delimiter"
    ] = (
        df[
            "delimiter_character"
        ]
        .notna()
    )

    last_delim = (
        delim_rows
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .max()
    )

    df[
        "final_header_line"
    ] = (
        df[
            "header_delimiter"
        ]
        &
        (
            df[
                "line_num"
            ]
            ==
            df[
                "page"
            ]
            .map(
                last_delim
            )
        )
    )

    chars = (
        df
        .loc[
            df[
                "header_delimiter"
            ],
            "delimiter_character"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    delimiter_char = (
        chars[0]
        if chars
        else None
    )

    delimiter_bool = bool(
        df[
            "header_delimiter"
        ].any()
    )

    type_attributes[
        "delimiter"
    ] = delimiter_bool

    type_attributes[
        "delimiter_char"
    ] = delimiter_char

    return (
        df,
        delimiter_bool,
        delimiter_char
    )



def header_lines(
    in_char_df,
    in_header_df
):
    """Identify likely header lines using form labels and table position."""

    horizontal_lines = []

    if in_header_df.empty:
        return horizontal_lines

    first_row = (
        in_header_df[
            "line_num"
        ].min()
    )

    first_chars = (
        in_char_df[
            in_char_df[
                "line_num"
            ]
            ==
            first_row
        ]
    )

    if not first_chars.empty:

        horizontal_lines.append(
            first_chars[
                "top"
            ].max()
        )

    final_headers = (
        in_header_df[
            in_header_df[
                "final_header_line"
            ]
            ==
            True
        ]
    )

    if not final_headers.empty:

        header_row = (
            final_headers[
                "line_num"
            ].min()
        )

        header_chars = (
            in_char_df[
                in_char_df[
                    "line_num"
                ]
                ==
                header_row
            ]
        )

        if not header_chars.empty:

            horizontal_lines.append(
                header_chars[
                    "bottom"
                ].min()
            )

    return horizontal_lines



def classify_headers_with_delimiters(
    in_line_df
):
    """Classify header rows when a reliable visual delimiter is available."""

    df = (
        in_line_df
        .sort_values(
            [
                "page",
                "line_num"
            ]
        )
        .copy()
    )

    df[
        "header"
    ] = False

    cutoff = (
        df.loc[
            df[
                "final_header_line"
            ],
            [
                "page",
                "line_num"
            ]
        ]
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .max()
    )

    mapped = (
        df[
            "page"
        ]
        .map(
            cutoff
        )
    )

    df[
        "header"
    ] = (
        mapped.notna()
        &
        (
            df[
                "line_num"
            ]
            <=
            mapped
        )
    )

    df = (
        df[
            ~(
                df[
                    "header"
                ]
                &
                df[
                    "header_delimiter"
                ]
            )
        ]
        .reset_index(
            drop=True
        )
    )

    return df.drop(
        columns=["data_anchor"],
        errors="ignore"
    )


def classify_headers_no_delimiters_collapsed(
    in_line_df
):
    """Classify headers for multi-line entities without a visual delimiter."""

    df = (
        in_line_df
        .sort_values(
            [
                "page",
                "line_num"
            ]
        )
        .copy()
    )

    df[
        "header"
    ] = False

    # EIN markers identify data rows even when values are blank.
    normalized_text = (
        df[
            "text"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    df[
        "data_anchor"
    ] = normalized_text.str.contains(
        EIN_OR_APPLIEDFOR_PAT,
        regex=True,
        na=False
    )

    first_anchor_line = (
        df.loc[
            df[
                "data_anchor"
            ],
            [
                "page",
                "line_num"
            ]
        ]
        .groupby(
            "page"
        )["line_num"]
        .min()
    )

    first_complete_line = (
        df.loc[
            df[
                "all_info"
            ],
            [
                "page",
                "line_num"
            ]
        ]
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .min()
    )

    first_data_line = first_anchor_line.combine_first(
        first_complete_line
    )

    mapped = (
        df[
            "page"
        ]
        .map(
            first_data_line
        )
    )

    df[
        "header"
    ] = (
        mapped.notna()
        &
        (
            df[
                "line_num"
            ]
            <
            mapped
        )
    )

    return df.drop(
        columns=["data_anchor"],
        errors="ignore"
    )


def classify_headers_no_delimiters_uncollapsed(
    in_line_df
):
    """Classify headers for one-line entities without a visual delimiter."""

    df = (
        in_line_df.copy()
    )

    df[
        "header"
    ] = False

    return df


def classify_headers_two_line_entities(in_line_df):
    """Mark headers while preserving name and identity lines of two-line entities."""
    """Keep the name line paired with the first P/S + EIN data line."""
    df = in_line_df.sort_values(["page", "line_num"]).copy()
    normalized = (
        df["text"].fillna("").astype(str).str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    identity = normalized.str.match(rf"^[SP][X*]*(?:{EIN_OR_APPLIEDFOR_PAT})")
    first_identity = df.loc[identity, ["page", "line_num"]].groupby("page")["line_num"].min()
    # The associated name is the line immediately before the P/S + EIN line.
    first_data = first_identity - 1
    df["header"] = df["line_num"] < df["page"].map(first_data).fillna(-1)
    return df


def remove_duplicate_headers(
    in_line_df
):
    """Remove repeated form headers that occur after page breaks."""

    return (
        in_line_df[
            ~(
                (
                    in_line_df[
                        "page"
                    ]
                    >
                    0
                )
                &
                (
                    in_line_df[
                        "header"
                    ]
                )
            )
        ]
        .reset_index(
            drop=True
        )
    )



def remove_totals(
    in_line_df
):
    """Exclude total and summary lines from the extracted entity data."""

    df = (
        in_line_df
        .sort_values(
            [
                "page",
                "line_num"
            ]
        )
        .copy()
    )

    if df.empty:
        return df

    last_page = (
        df[
            "page"
        ].max()
    )

    second_last_page = (
        last_page - 1
    )

    last_line_per_page = (
        df
        .groupby(
            "page"
        )[
            "line_num"
        ]
        .transform(
            "max"
        )
    )

    df[
        "is_total_cutoff"
    ] = (
        df[
            "text"
        ]
        .astype(str)
        .str[:5]
        .str.upper()
        .eq(
            "TOTAL"
        )
        &
        (
            (
                df[
                    "page"
                ]
                ==
                last_page
            )
            |
            (
                (
                    df[
                        "page"
                    ]
                    ==
                    second_last_page
                )
                &
                (
                    df[
                        "line_num"
                    ]
                    ==
                    last_line_per_page
                )
            )
        )
    )

    cut = (
        df.loc[
            df[
                "is_total_cutoff"
            ],
            [
                "page",
                "line_num"
            ]
        ]
        .head(1)
    )

    if cut.empty:

        return df.drop(
            columns=[
                "is_total_cutoff"
            ]
        )

    cut_page = (
        cut[
            "page"
        ].iloc[0]
    )

    cut_line = (
        cut[
            "line_num"
        ].iloc[0]
    )

    df = (
        df[
            (
                df[
                    "page"
                ]
                <
                cut_page
            )
            |
            (
                (
                    df[
                        "page"
                    ]
                    ==
                    cut_page
                )
                &
                (
                    df[
                        "line_num"
                    ]
                    <
                    cut_line
                )
            )
        ]
        .reset_index(
            drop=True
        )
    )

    return df.drop(
        columns=[
            "is_total_cutoff"
        ]
    )



def classify_noncollapsed_rows(
    in_line_df
):
    """Label identity, name, and numeric components of one-line data rows."""

    df = (
        in_line_df.copy()
    )

    def classify_text(text):
        """Identify which expected Schedule E components occur in one text line."""

        original = re.sub(
            r"\s+",
            "",
            str(text).upper()
        )

        s = original

        out = {
            "contains_ein": False,
            "contains_value": False,
            "contains_P_S": False,
            "contains_name": False,
            "text_remaining": original
        }

        if re.search(
            EIN_OR_APPLIEDFOR_PAT,
            s
        ):

            out[
                "contains_ein"
            ] = True

            s = re.sub(
                EIN_OR_APPLIEDFOR_PAT,
                "",
                s
            )

        if re.search(
            PS_LEAD_PAT,
            s
        ):

            out[
                "contains_P_S"
            ] = True

            s = re.sub(
                PS_LEAD_PAT,
                "",
                s,
                count=1
            )

        if re.search(
            VALUE_PAT,
            s
        ):

            out[
                "contains_value"
            ] = True

            s = re.sub(
                VALUE_PAT,
                "",
                s
            )

        remaining = re.sub(
            r"[^A-Z0-9]+",
            "",
            s
        )

        if remaining != "":

            out[
                "contains_name"
            ] = True

        out[
            "text_remaining"
        ] = s

        return pd.Series(
            out
        )

    classified = (
        df[
            "text"
        ]
        .apply(
            classify_text
        )
    )

    return pd.concat(
        [
            df,
            classified
        ],
        axis=1
    )



def collapse_rows(
    in_line_df
):
    """Combine related text lines into logical Schedule E entity rows."""

    df = (
        in_line_df
        .sort_values(
            [
                "page",
                "line_num"
            ]
        )
        .copy()
    )

    out_rows = []

    def append_component(
        block,
        text,
        header,
        name_ind=0,
        ein_ind=0,
        other_ind=0
    ):
        """Append a classified line fragment to the active logical entity."""

        if (
            block.empty
            or
            text == ""
        ):
            return

        block = (
            block
            .sort_values(
                "line_num"
            )
        )

        x1_sum = (
            pd.to_numeric(
                block[
                    "x1"
                ],
                errors="coerce"
            )
            .sum(
                skipna=True
            )
        )

        out_rows.append({
            "page":
                block[
                    "page"
                ].iloc[0],

            "line_num":
                block[
                    "line_num"
                ].iloc[0],

            "last_line_num":
                block[
                    "line_num"
                ].iloc[-1],

            "x0":
                block[
                    "x0"
                ].iloc[0],

            "x1":
                float(
                    x1_sum
                ),

            "text":
                text,

            "header":
                header,

            "name_ind":
                name_ind,

            "ein_ind":
                ein_ind,

            "value_ind":
                other_ind
        })

    def finalize_entity(
        block
    ):
        """Finalize the active entity and append it when it contains data."""

        if block.empty:
            return

        block = (
            block
            .sort_values(
                "line_num"
            )
            .reset_index(
                drop=True
            )
        )

        ein_mask = (
            block[
                "contains_ein"
            ]
            .fillna(
                False
            )
        )

        # Continue the current entity until an EIN is found.

        if not ein_mask.any():

            tail_mask = (
                block[
                    "contains_P_S"
                ]
                .fillna(
                    False
                )
                |
                block[
                    "contains_value"
                ]
                .fillna(
                    False
                )
            )

            if tail_mask.any():

                first_tail_idx = (
                    tail_mask.idxmax()
                )

                if first_tail_idx > 0:

                    name_block = (
                        block.loc[
                            :first_tail_idx - 1
                        ]
                    )

                else:

                    name_block = (
                        block.iloc[
                            0:0
                        ]
                    )

                other_block = (
                    block.loc[
                        first_tail_idx:
                    ]
                    .copy()
                )

                if not name_block.empty:

                    append_component(
                        name_block,
                        "".join(
                            name_block[
                                "text"
                            ]
                            .astype(str)
                        ),
                        header=False,
                        name_ind=1
                    )

                if not other_block.empty:

                    other_text = "".join(
                        re.sub(
                            r"\s+",
                            "",
                            str(t).upper()
                        )
                        for t in other_block[
                            "text"
                        ]
                    )

                    append_component(
                        other_block,
                        other_text,
                        header=False,
                        other_ind=1
                    )

                return

            append_component(
                block,
                "".join(
                    block[
                        "text"
                    ]
                    .astype(str)
                ),
                header=False,
                name_ind=1
            )

            return

        # Start a new entity at each EIN.

        first_ein_idx = (
            ein_mask.idxmax()
        )

        if first_ein_idx > 0:

            name_block = (
                block.loc[
                    :first_ein_idx - 1
                ]
            )

        else:

            name_block = (
                block.iloc[
                    0:0
                ]
            )

        if not name_block.empty:

            append_component(
                name_block,
                "".join(
                    name_block[
                        "text"
                    ]
                    .astype(str)
                ),
                header=False,
                name_ind=1
            )

        ein_block = (
            block[
                block[
                    "contains_ein"
                ]
                .fillna(
                    False
                )
            ]
            .copy()
        )

        if not ein_block.empty:

            ein_text = "".join(
                "".join(
                    re.findall(
                        EIN_PAT,
                        re.sub(
                            r"\s+",
                            "",
                            str(t).upper()
                        )
                    )
                )
                for t in ein_block[
                    "text"
                ]
            )

            append_component(
                ein_block,
                ein_text,
                header=False,
                ein_ind=1
            )

        tail_block = (
            block.loc[
                first_ein_idx:
            ]
            .copy()
        )

        other_parts = []
        other_rows_idx = []

        for idx, row in tail_block.iterrows():

            s = re.sub(
                r"\s+",
                "",
                str(
                    row[
                        "text"
                    ]
                ).upper()
            )

            s_without_ein = re.sub(
                EIN_PAT,
                "",
                s
            )

            if s_without_ein != "":

                other_parts.append(
                    s_without_ein
                )

                other_rows_idx.append(
                    idx
                )

        if other_parts:

            other_block = (
                tail_block.loc[
                    other_rows_idx
                ]
            )

            append_component(
                other_block,
                "".join(
                    other_parts
                ),
                header=False,
                other_ind=1
            )


    for page, group in df.groupby(
        "page",
        sort=False
    ):

        group = (
            group
            .sort_values(
                "line_num"
            )
            .reset_index(
                drop=True
            )
        )

        header_block = (
            group[
                group[
                    "header"
                ]
            ]
            .copy()
        )

        if not header_block.empty:

            append_component(
                header_block,
                "".join(
                    header_block[
                        "text"
                    ]
                    .astype(str)
                ),
                header=True
            )

        group = (
            group[
                ~group[
                    "header"
                ]
            ]
            .reset_index(
                drop=True
            )
        )

        if group.empty:
            continue

        current_rows = []
        seen_tail = False

        for _, row in group.iterrows():

            is_name_only = (
                bool(
                    row.get(
                        "contains_name",
                        False
                    )
                )
                and
                not bool(
                    row.get(
                        "contains_ein",
                        False
                    )
                )
                and
                not bool(
                    row.get(
                        "contains_P_S",
                        False
                    )
                )
                and
                not bool(
                    row.get(
                        "contains_value",
                        False
                    )
                )
            )

            has_tail_info = (
                bool(
                    row.get(
                        "contains_ein",
                        False
                    )
                )
                or
                bool(
                    row.get(
                        "contains_P_S",
                        False
                    )
                )
                or
                bool(
                    row.get(
                        "contains_value",
                        False
                    )
                )
            )

            if not current_rows:

                current_rows.append(
                    row
                )

                seen_tail = (
                    has_tail_info
                )

                continue

            if (
                is_name_only
                and
                seen_tail
            ):

                finalize_entity(
                    pd.DataFrame(
                        current_rows
                    )
                )

                current_rows = [
                    row
                ]

                seen_tail = False

            else:

                current_rows.append(
                    row
                )

                seen_tail = (
                    seen_tail
                    or
                    has_tail_info
                )

        if current_rows:

            finalize_entity(
                pd.DataFrame(
                    current_rows
                )
            )

    return pd.DataFrame(
        out_rows,
        columns=[
            "page",
            "line_num",
            "last_line_num",
            "x0",
            "x1",
            "text",
            "header",
            "name_ind",
            "ein_ind",
            "value_ind"
        ]
    )
