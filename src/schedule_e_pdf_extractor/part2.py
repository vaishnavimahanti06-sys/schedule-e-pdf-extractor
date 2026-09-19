"""Schedule E table reconstruction, validation, and batch processing."""

from .part1 import *



def shift_rows(in_line_df):
    """Align name and EIN lines to the same reconstructed table-row geometry."""

    line_df = in_line_df.copy()

    line_df["x0_new"] = line_df["x0"]
    line_df["x1_new"] = line_df["x1"]

    non_header = ~line_df["header"]

    ein_mask = (
        non_header
        & (line_df["ein_ind"] == 1)
    )

    name_mask = (
        non_header
        & (line_df["name_ind"] == 1)
    )

    max_ein = pd.to_numeric(
        line_df.loc[ein_mask, "x1"],
        errors="coerce"
    ).max()

    min_ein = pd.to_numeric(
        line_df.loc[ein_mask, "x0"],
        errors="coerce"
    ).min()

    min_name = pd.to_numeric(
        line_df.loc[name_mask, "x0"],
        errors="coerce"
    ).min()

    min_all = pd.Series([
        min_ein,
        min_name
    ]).min()

    if pd.isna(min_all) or pd.isna(max_ein):
        return line_df, None, None

    row_width = pd.to_numeric(
        line_df["x1"],
        errors="coerce"
    )

    line_df.loc[ein_mask, "x1_new"] = float(min_all)

    line_df.loc[ein_mask, "x0_new"] = (
        min_all - row_width.loc[ein_mask]
    )

    right_name = min_all - max_ein

    line_df.loc[name_mask, "x1_new"] = float(right_name)

    line_df.loc[name_mask, "x0_new"] = (
        right_name - row_width.loc[name_mask]
    )

    return line_df, min_all, max_ein



def join_to_char_df_uncollapsed(
    in_line_df,
    in_char_df
):
    """Map character geometry into logical rows when entities span source lines."""

    line_df = in_line_df.copy()
    char_df = in_char_df.copy()

    required_line_cols = {
        "page",
        "line_num",
        "last_line_num",
        "header",
        "name_ind",
        "ein_ind",
        "value_ind",
        "x0_new",
        "x1_new"
    }
    missing_columns = required_line_cols - set(line_df.columns)
    if missing_columns:
        raise ValueError(
            "join_to_char_df_uncollapsed received invalid line data. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    char_df["char_width"] = (
        char_df["x1"] - char_df["x0"]
    )

    line_df["comb_line_num"] = pd.NA

    start_mask = (
        ~line_df["header"]
        & (line_df["name_ind"] == 1)
    )

    line_df.loc[start_mask, "comb_line_num"] = (
        line_df.loc[start_mask, "line_num"]
    )

    line_df["comb_line_num"] = (
        line_df
        .groupby("page", sort=False)["comb_line_num"]
        .ffill()
    )

    line_df["last_line_num"] = (
        line_df["last_line_num"]
        .fillna(line_df["line_num"])
    )

    char_min_x0 = (
        char_df
        .groupby(
            ["page", "line_num"],
            as_index=False
        )
        .agg(
            char_min_x0=("x0", "min")
        )
    )

    line_df = (
        line_df
        .merge(
            char_min_x0,
            on=["page", "line_num"],
            how="inner"
        )
    )

    line_df["x_shift"] = (
        line_df["x0_new"]
        - line_df["char_min_x0"]
    )

    map_df = line_df[
        [
            "page",
            "line_num",
            "last_line_num",
            "header",
            "x_shift",
            "comb_line_num",
            "name_ind",
            "ein_ind",
            "value_ind"
        ]
    ].copy()

    map_df = map_df.rename(
        columns={
            "line_num": "line_num_start"
        }
    )

    map_df["span_len"] = (
        map_df["last_line_num"]
        - map_df["line_num_start"]
    )

    map_df["line_num_right"] = map_df.apply(
        lambda r: list(
            range(
                int(r["line_num_start"]),
                int(r["last_line_num"]) + 1
            )
        ),
        axis=1
    )

    map_df = (
        map_df
        .explode(
            "line_num_right",
            ignore_index=True
        )
    )

    map_df = (
        map_df
        .sort_values(
            [
                "page",
                "line_num_right",
                "span_len",
                "line_num_start"
            ],
            ascending=[
                True,
                True,
                False,
                True
            ]
        )
        .drop_duplicates(
            subset=[
                "page",
                "line_num_right"
            ],
            keep="first"
        )
    )

    char_df = (
        char_df
        .merge(
            map_df[
                [
                    "page",
                    "line_num_right",
                    "header",
                    "x_shift",
                    "comb_line_num",
                    "last_line_num",
                    "name_ind",
                    "ein_ind",
                    "value_ind"
                ]
            ],
            left_on=[
                "page",
                "line_num"
            ],
            right_on=[
                "page",
                "line_num_right"
            ],
            how="inner"
        )
    )

    non_header = ~char_df["header"]

    concat_mask = (
        non_header
        & char_df["comb_line_num"].notna()
        & char_df["name_ind"].fillna(0).eq(1)
    )

    char_df.loc[non_header, "x0"] = (
        char_df.loc[non_header, "x0"]
        + char_df.loc[non_header, "x_shift"]
    )

    char_df.loc[non_header, "x1"] = (
        char_df.loc[non_header, "x1"]
        + char_df.loc[non_header, "x_shift"]
    )

    mean_char = char_df.loc[
        non_header,
        "char_width"
    ].mean()

    grp_cols = [
        "page",
        "comb_line_num"
    ]

    line_stats = (
        char_df.loc[concat_mask]
        .groupby(
            grp_cols + ["line_num"],
            as_index=False
        )
        .agg(
            min_x0=("x0", "min"),
            max_x1=("x1", "max")
        )
    )

    def _calc_concat_offsets(group):
        """Calculate offsets that place line fragments consecutively in one row."""

        comb_ln = group["comb_line_num"].iloc[0]

        group = (
            group
            .assign(
                _anchor=(
                    group["line_num"] != comb_ln
                ).astype(int)
            )
            .sort_values(
                [
                    "_anchor",
                    "line_num"
                ]
            )
            .copy()
        )

        offsets = []
        current_end = None

        for i, row in enumerate(
            group.itertuples(index=False)
        ):

            if i == 0:

                offset = 0.0
                current_end = row.max_x1

            else:

                offset = (
                    current_end
                    + mean_char
                    - row.min_x0
                )

                current_end = (
                    row.max_x1
                    + offset
                )

            offsets.append(offset)

        group["x_concat_shift"] = offsets

        return group.drop(
            columns=["_anchor"]
        )

    if not line_stats.empty:

        offset_groups = [
            _calc_concat_offsets(group)
            for _, group in line_stats.groupby(grp_cols, sort=False)
        ]
        concat_offsets = pd.concat(offset_groups, ignore_index=True)

        char_df = (
            char_df
            .merge(
                concat_offsets[
                    grp_cols
                    + [
                        "line_num",
                        "x_concat_shift"
                    ]
                ],
                on=grp_cols + ["line_num"],
                how="left"
            )
        )

        shift_mask = (
            concat_mask
            & char_df["x_concat_shift"].notna()
        )

        char_df.loc[
            shift_mask,
            "x0"
        ] += char_df.loc[
            shift_mask,
            "x_concat_shift"
        ]

        char_df.loc[
            shift_mask,
            "x1"
        ] += char_df.loc[
            shift_mask,
            "x_concat_shift"
        ]

    else:

        char_df["x_concat_shift"] = np.nan

    anchor_chars = char_df.loc[
        concat_mask
        & (
            char_df["line_num"]
            == char_df["comb_line_num"]
        )
    ].copy()

    if not anchor_chars.empty:

        anchor_stats = (
            anchor_chars
            .groupby(
                [
                    "page",
                    "comb_line_num"
                ],
                as_index=False
            )
            .agg(
                anchor_bottom=("bottom", "min"),
                anchor_top=("top", "max")
            )
        )

        char_df = (
            char_df
            .merge(
                anchor_stats,
                on=[
                    "page",
                    "comb_line_num"
                ],
                how="left"
            )
        )

        char_df.loc[
            concat_mask,
            "bottom"
        ] = char_df.loc[
            concat_mask,
            "anchor_bottom"
        ]

        char_df.loc[
            concat_mask,
            "top"
        ] = char_df.loc[
            concat_mask,
            "anchor_top"
        ]

    char_df["char_width"] = (
        char_df["x1"] - char_df["x0"]
    )

    char_df = (
        char_df
        .drop(
            columns=[
                "line_num_right",
                "x_shift",
                "x_concat_shift",
                "anchor_bottom",
                "anchor_top"
            ],
            errors="ignore"
        )
    )

    return char_df, line_df



def join_to_char_df_normal(
    in_line_df,
    in_char_df
):
    """Join normal one-line row metadata back to the source characters."""

    line_df = in_line_df.copy()
    char_df = in_char_df.copy()

    char_bounds = (
        char_df
        .groupby(
            [
                "page",
                "line_num"
            ],
            as_index=False
        )
        .agg(
            x0_line=("x0", "min"),
            x1_line=("x1", "max")
        )
    )

    line_df = (
        line_df
        .merge(
            char_bounds,
            on=[
                "page",
                "line_num"
            ],
            how="inner"
        )
    )

    line_df["x0"] = line_df["x0_line"]
    line_df["x1"] = line_df["x1_line"]

    line_df = (
        line_df
        .drop(
            columns=[
                "x0_line",
                "x1_line"
            ]
        )
    )

    char_df = (
        char_df
        .merge(
            line_df[
                [
                    "page",
                    "line_num",
                    "header"
                ]
            ],
            on=[
                "page",
                "line_num"
            ],
            how="inner"
        )
    )

    char_df["char_width"] = (
        char_df["x1"] - char_df["x0"]
    )

    return char_df, line_df


def join_two_line_entity_chars(in_line_df, in_char_df):
    """Merge a name line with its following P/S + EIN line without shifting it."""
    line_df = in_line_df.sort_values(["page", "line_num"]).copy()
    normalized = (
        line_df["text"].fillna("").astype(str).str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    identity = normalized.str.match(
        rf"^[SP][X*]*(?:{EIN_OR_APPLIEDFOR_PAT})"
    )
    line_df["identity_line"] = identity
    line_df["logical_line_num"] = line_df["line_num"]
    previous_line = line_df.groupby("page", sort=False)["line_num"].shift(1)
    line_df.loc[identity, "logical_line_num"] = previous_line.loc[identity]

    mapping = line_df[
        [
            "page",
            "line_num",
            "logical_line_num",
            "header",
            "identity_line",
        ]
    ]
    char_df = in_char_df.merge(
        mapping,
        on=["page", "line_num"],
        how="inner",
    )
    char_df = char_df.loc[~char_df["header"].fillna(False)].copy()

    # Align below-name P/S and EIN glyphs with their table columns.
    identity_chars = char_df.loc[char_df["identity_line"]].copy()
    value_x0 = identity_chars.loc[
        identity_chars["x0"] > 1000,
        "x0",
    ]
    if not value_x0.empty:
        first_value_x0 = float(value_x0.min())
        ps_start = first_value_x0 - 520.0
        ein_start = first_value_x0 - 300.0
        for _, group in identity_chars.groupby(["page", "line_num"], sort=False):
            left_group = group.loc[group["x0"] < 1000].sort_values("x0")
            if len(left_group) < 2:
                continue
            ps_index = left_group.index[0]
            ein_indexes = left_group.index[1:]
            char_df.loc[ps_index, ["x0", "x1"]] = [
                ps_start,
                ps_start + (char_df.loc[ps_index, "x1"] - char_df.loc[ps_index, "x0"]),
            ]
            ein_original_x0 = float(char_df.loc[ein_indexes, "x0"].min())
            width = char_df.loc[ein_indexes, "x1"] - char_df.loc[ein_indexes, "x0"]
            new_x0 = ein_start + (char_df.loc[ein_indexes, "x0"] - ein_original_x0)
            char_df.loc[ein_indexes, "x0"] = new_x0
            char_df.loc[ein_indexes, "x1"] = new_x0 + width

    char_df["line_num"] = char_df["logical_line_num"].astype(int)
    char_df["char_width"] = char_df["x1"] - char_df["x0"]

    data_line_df = line_df.loc[~line_df["header"].fillna(False)].copy()
    return char_df, data_line_df



def words(in_char_df):
    """Group positioned OCR characters into words for column assignment."""

    df = in_char_df.copy()

    mean_char = df["char_width"].mean()

    if pd.isna(mean_char) or mean_char == 0:
        mean_char = 3.0

    if "comb_line_num" in df.columns:
        df["line_num"] = (
            df["comb_line_num"]
            .fillna(df["line_num"])
        )

    df = (
        df
        .sort_values(
            [
                "page",
                "line_num",
                "x0"
            ]
        )
        .reset_index(drop=True)
    )

    df["prev_x1"] = (
        df
        .groupby(
            [
                "page",
                "line_num"
            ]
        )["x1"]
        .shift(1)
    )

    df["gap"] = (
        df["x0"] - df["prev_x1"]
    )

    df["space"] = (
        df["gap"]
        .abs()
        .gt(0.5 * mean_char)
        & ~df["header"]
    )

    df["word_id"] = (
        df
        .groupby(
            [
                "page",
                "line_num"
            ],
            sort=False
        )["space"]
        .cumsum()
        .fillna(0)
        .astype("int64")
    )

    word_df = (
        df
        .groupby(
            [
                "page",
                "line_num",
                "word_id"
            ],
            as_index=False
        )
        .agg(
            x0=("x0", "min"),
            x1=("x1", "max"),
            text=("text", "".join),
            header=("header", "any"),
            prev_x1=("prev_x1", "first")
        )
    )

    return word_df



def columns_header_delimiter(
    in_char_df,
    in_header_df
):
    """Infer column boundaries from spacing in the final header line."""

    final_headers = (
        in_header_df[
            in_header_df["final_header_line"] == True
        ]
    )

    if final_headers.empty:
        return []

    header_page = final_headers["page"].iloc[0]
    header_line_num = final_headers["line_num"].iloc[0]

    header_char_line = (
        in_char_df[
            (in_char_df["page"] == header_page)
            & (in_char_df["line_num"] == header_line_num)
        ]
        .copy()
    )

    if header_char_line.empty:
        return []

    header_char_line = (
        header_char_line
        .sort_values("x0")
        .reset_index(drop=True)
    )

    header_char_line["char_width"] = (
        header_char_line["x1"]
        - header_char_line["x0"]
    )

    mean_char = header_char_line["char_width"].mean()

    header_char_line["prev_x1"] = (
        header_char_line["x1"].shift(1)
    )

    header_char_line["gap"] = (
        header_char_line["x0"]
        - header_char_line["prev_x1"]
    )

    vertical_lines = (
        header_char_line[
            header_char_line["gap"] > 0.9 * mean_char
        ]["x0"]
        .tolist()
    )

    return sorted(set(vertical_lines))



def columns_words_no_delimiter(
    in_char_df,
    in_line_df,
    num_cols=None
):
    """Dynamically detect table columns from P/S, EIN, and value geometry."""
    required_char = {"page", "line_num", "text", "x0", "x1", "top", "bottom"}
    required_line = {"page", "line_num", "text"}
    missing_char = required_char - set(in_char_df.columns)
    missing_line = required_line - set(in_line_df.columns)
    if missing_char or missing_line:
        raise ValueError(f"Dynamic column detection missing fields: chars={sorted(missing_char)}, lines={sorted(missing_line)}")

    work = in_char_df.copy()
    work["text"] = work["text"].fillna("").astype(str)
    work = work.loc[work["text"].str.strip().ne("")].copy()
    if "comb_line_num" in work:
        work["line_num"] = work["comb_line_num"].fillna(work["line_num"])
    work = work.sort_values(["page", "line_num", "x0", "x1"])
    work["prev_x1"] = work.groupby(["page", "line_num"], sort=False)["x1"].shift(1)
    widths = work["x1"] - work["x0"]
    widths = widths[np.isfinite(widths) & widths.gt(0)]
    char_width = float(widths.median()) if not widths.empty else 4.0
    # OCR-generated searchable PDFs commonly encode a word space at only
    # slightly more than one character width. Preserve those name-word gaps.
    word_gap_threshold = max(char_width * 1.1, 2.0)
    work["new_word"] = work["prev_x1"].isna() | ((work["x0"] - work["prev_x1"]) > word_gap_threshold)
    work["word_id"] = work.groupby(["page", "line_num"], sort=False)["new_word"].cumsum()
    words_df = work.groupby(["page", "line_num", "word_id"], sort=False, as_index=False).agg(text=("text", "".join), x0=("x0", "min"), x1=("x1", "max"), top=("top", "min"), bottom=("bottom", "max"))
    words_df["text"] = words_df["text"].str.strip()

    lines_df = in_line_df.copy()
    if "header" in lines_df:
        lines_df = lines_df.loc[~lines_df["header"].fillna(False)]
    words_df = words_df.merge(lines_df[["page", "line_num"]].drop_duplicates(), on=["page", "line_num"], how="inner")
    if words_df.empty:
        raise ValueError("Dynamic column detection found no data-row words.")
    words_df["norm"] = words_df["text"].map(lambda value: re.sub(r"\s+", "", str(value).upper()))
    def positions(pattern):
        """Return horizontal centers for words matching an anchor pattern."""
        matched = words_df["norm"].str.fullmatch(pattern, na=False)
        return ((words_df.loc[matched, "x0"] + words_df.loc[matched, "x1"]) / 2.0).tolist()
    def center(values):
        """Return the median horizontal center for a set of anchor positions."""
        return float(np.median(values)) if values else None
    def clusters(values):
        """Cluster nearby anchor positions into distinct physical columns."""
        if not values:
            return []
        tolerance = max(char_width * 4.0, 10.0)
        groups = [[value] for value in sorted(map(float, values))]
        result = [groups[0]]
        for group in groups[1:]:
            if abs(group[0] - np.median(result[-1])) <= tolerance:
                result[-1].extend(group)
            else:
                result.append(group)
        return [float(np.median(group)) for group in result]
    ps_center = center(positions(r"P|S|PX+|SX+|\*+"))
    ein_center = center(positions(EIN_OR_APPLIEDFOR_PAT))
    numeric_centers = clusters(positions(VALUE_PAT))
    if ein_center is not None:
        numeric_centers = [value for value in numeric_centers if abs(value - ein_center) > max(char_width * 6, 20)]
    if not numeric_centers:
        raise ValueError("Dynamic column detection found no numeric/value anchors.")
    left, right = float(words_df["x0"].min()), float(words_df["x1"].max())
    spacing = max((right - left) * 0.07, char_width * 12)
    if len(numeric_centers) > 1:
        gaps = np.diff(numeric_centers)
        valid = gaps[gaps > max(char_width * 4, 10)]
        if len(valid):
            spacing = float(np.median(valid))
    if len(numeric_centers) == 1:
        available = right - (ein_center if ein_center is not None else left + (right - left) * 0.55)
        count = min(20, max(1, int(round(available / spacing))))
        numeric_centers = [numeric_centers[0] + index * spacing for index in range(count) if numeric_centers[0] + index * spacing <= right]
    if ps_center is None:
        ps_center = ein_center - max(char_width * 10, 35) if ein_center is not None else numeric_centers[0] - max(char_width * 18, 70)
    if ein_center is None:
        ein_center = (ps_center + numeric_centers[0]) / 2.0
    # Keep the narrow P/S token with the identity columns.
    boundaries = [ps_center - max(char_width * 2.0, 10.0), (ps_center + ein_center) / 2.0, (ein_center + numeric_centers[0]) / 2.0]
    boundaries.extend((a + b) / 2.0 for a, b in zip(numeric_centers[:-1], numeric_centers[1:]))
    boundaries = sorted(set(map(float, boundaries)))
    print("Schedule E dynamic column detection")
    print(f"  P/S center: {ps_center:.2f}; EIN center: {ein_center:.2f}")
    print("  Numeric centers:", [round(value, 2) for value in numeric_centers])
    print("  Boundaries:", [round(value, 2) for value in boundaries])
    print(f"  Detected columns: {len(boundaries) + 1}")
    return words_df.drop(columns="norm"), boundaries



def assign_words_to_columns(
    words_df,
    boundaries
):
    """Assign each word to a dynamic column using its horizontal center."""
    if not boundaries:
        raise ValueError("assign_words_to_columns: no boundaries supplied.")
    boundaries = np.asarray(boundaries, dtype=float)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("assign_words_to_columns: boundaries must increase.")
    word_df = words_df.copy()
    word_df["x_center"] = (word_df["x0"] + word_df["x1"]) / 2.0
    word_df["column"] = np.searchsorted(boundaries, word_df["x_center"].to_numpy(), side="right")
    return word_df



def reconstitute_table(
    in_word_df,
    num_expected_cols=None
):
    """Pivot assigned words into ordered rows with dynamically detected columns."""

    df = in_word_df.copy()

    if df.empty:
        raise ValueError("reconstitute_table: no words available.")

    required = {"page", "line_num", "column", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reconstitute_table: missing columns {sorted(missing)}")

    if "header" in df.columns:
        df = df.loc[~df["header"].fillna(False)].copy()
    if df.empty:
        raise ValueError("reconstitute_table: no data rows remain.")

    df["column"] = (
        pd.to_numeric(
            df["column"],
            errors="coerce"
        )
        .astype("Int64")
    )

    sort_cols = [
        c
        for c in [
            "page",
            "line_num",
            "column",
            "x0"
        ]
        if c in df.columns
    ]

    df = (
        df
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    cell_df = (
        df
        .groupby(
            [
                "page",
                "line_num",
                "column"
            ],
            as_index=False
        )
        .agg(
            text=(
                "text",
                lambda s: " ".join(
                    [
                        t
                        for t in s.astype(str)
                        if t and t != "nan"
                    ]
                )
            )
        )
    )

    detected_cols = int(df["column"].max()) + 1
    num_expected_cols = detected_cols if num_expected_cols is None else max(num_expected_cols, detected_cols)
    table = cell_df.pivot_table(
        index=["page", "line_num"],
        columns="column",
        values="text",
        aggfunc="first"
    )
    table = table.reindex(columns=range(num_expected_cols)).reset_index()
    table.columns = ["page", "line_num", *range(num_expected_cols)]
    return table.sort_values(["page", "line_num"]).reset_index(drop=True)


def merge_wrapped_entities(table_df):
    """Attach name-only OCR continuation lines to their following data row."""
    df = table_df.copy()
    data_columns = [
        column
        for column in df.columns
        if isinstance(column, (int, np.integer))
    ]
    if len(data_columns) < 3:
        return df

    name_col, ps_col, ein_col = data_columns[:3]
    value_cols = data_columns[3:]
    pending_names = []
    output_rows = []

    def cell_text(value):
        """Convert a potentially missing table cell to normalized display text."""
        if pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "<na>"} else text

    for _, row in df.iterrows():
        name = cell_text(row[name_col])
        ps = cell_text(row[ps_col])
        ein = cell_text(row[ein_col])
        has_ps_anchor = bool(re.fullmatch(r"[PS](?:X+)?", ps))
        has_ein_anchor = bool(re.fullmatch(EIN_OR_APPLIEDFOR_PAT, ein))
        # P/S and EIN anchors identify a new logical row.
        if name and not has_ps_anchor and not has_ein_anchor:
            if output_rows:
                previous = output_rows[-1].copy()
                previous_ps = cell_text(previous[ps_col])
                previous_ein = cell_text(previous[ein_col])
                if (
                    re.fullmatch(r"[PS](?:X+)?", previous_ps)
                    and re.fullmatch(EIN_OR_APPLIEDFOR_PAT, previous_ein)
                ):
                    previous[name_col] = " ".join(
                        part for part in (cell_text(previous[name_col]), name)
                        if part
                    )
                    output_rows[-1] = previous
                    continue
            pending_names.append(" ".join(part for part in (name, ps, ein) if part))
            continue

        if pending_names:
            row = row.copy()
            row[name_col] = " ".join([*pending_names, name]).strip()
            pending_names.clear()
        output_rows.append(row)

    if pending_names:
        for name in pending_names:
            orphan = {column: "" for column in df.columns}
            orphan[name_col] = name
            output_rows.append(pd.Series(orphan))

    return pd.DataFrame(output_rows, columns=df.columns).reset_index(drop=True)



def clean_numeric_value(value):
    """Normalize a currency-like OCR value to an integer, defaulting invalid text to zero."""

    if pd.isna(value):
        return 0

    s = str(value).strip()

    if s == "" or s.upper() == "NONE":
        return 0

    negative = (
        s.startswith("(")
        and s.endswith(")")
    )

    s = (
        s
        .replace("$", "")
        .replace(",", "")
        .replace(" ", "")
    )

    if negative:
        s = "-" + s[1:-1]

    if (
        s.endswith(".")
        and s.count(".") == 1
    ):
        s = s[:-1]

    try:
        return int(float(s))

    except Exception:
        return 0



def table_cleaning(in_table_df):
    """Apply Schedule E headers, normalize values, and remove reconstruction artifacts."""

    df = in_table_df.copy()

    if df.empty:
        return df

    # Remove reconstruction metadata from the exported table.
    df = df.drop(columns=[column for column in ("page", "line_num") if column in df.columns])

    new_cols = [
        f"col{i + 1}"
        for i in range(df.shape[1])
    ]

    new_cols[0] = "NAME"
    if len(new_cols) > 1:
        new_cols[1] = "P/S"
    if len(new_cols) > 2:
        new_cols[2] = "EIN"

    df.columns = new_cols

    df["NAME"] = (
        df["NAME"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\(continued\s*on\s*next\s*page\).*?Page\s*\d+\s*",
            "",
            regex=True,
            flags=re.IGNORECASE
        )
        .str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True)
        .str.replace(r"\bSCorp\b", "S Corp", regex=True)
        .str.replace(r"\bSCorporation\b", "S Corporation", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if "P/S" in df.columns:
        ps_values = df["P/S"].fillna("").astype(str).str.strip()
        normalized_ps = (
            ps_values
            .str.extract(r"^([PS])", flags=re.IGNORECASE, expand=False)
            .str.upper()
        )
        df["P/S"] = normalized_ps.fillna(ps_values)

    if df.shape[1] >= 6:

        last5 = list(
            df.columns[-5:]
        )

        tail = df[last5]

        populated = (
            tail
            .astype(str)
            .apply(
                lambda col:
                col.str.strip().ne("")
            )
            .to_numpy()
            .any()
        )

        if populated:

            rename_map = dict(
                zip(
                    last5,
                    VALUE_LABELS
                )
            )

            df = df.rename(
                columns=rename_map
            )

    for col in VALUE_LABELS:

        if col in df.columns:

            df[col] = (
                df[col]
                .apply(clean_numeric_value)
            )

    # Remove blank separator rows without removing valid zero-value entities.
    identifier_cols = [
        col
        for col in ("NAME", "P/S", "EIN")
        if col in df.columns
    ]
    numeric_cols = [
        col
        for col in VALUE_LABELS
        if col in df.columns
    ]
    if identifier_cols and numeric_cols:
        empty_identifiers = (
            df[identifier_cols]
            .fillna("")
            .astype(str)
            .apply(
                lambda col: col.str.strip().isin(
                    ("", "nan", "None", "<NA>")
                )
            )
            .all(axis=1)
        )
        zero_values = df[numeric_cols].eq(0).all(axis=1)
        df = df.loc[~(empty_identifiers & zero_values)].reset_index(drop=True)

        # A repeated form header can leave a text fragment such as
        # "Company S Corp" in the name column. It cannot represent a
        # Schedule E entity without both P/S and EIN, and all value cells
        # are blank/zero, so exclude it before confidence scoring.
        if {"P/S", "EIN"}.issubset(df.columns):
            zero_values = df[numeric_cols].eq(0).all(axis=1)
            missing_identity = (
                df["P/S"].fillna("").astype(str).str.strip().eq("")
                & df["EIN"].fillna("").astype(str).str.strip().eq("")
            )
            df = df.loc[~(missing_identity & zero_values)].reset_index(drop=True)

    if "EIN" in df.columns:
        df["EIN"] = df["EIN"].replace("APPLIEDFOR", "APPLIED FOR")

    return df



def extraction_confidence(df):
    """Score whether extracted rows contain the identities and values expected in Schedule E."""

    if df is None or df.empty:
        return 0.0

    score = 0.0

    if "NAME" in df.columns:

        valid_names = (
            df["NAME"]
            .astype(str)
            .str.strip()
            .ne("")
            .mean()
        )

        score += 0.25 * valid_names

    combined = df.apply(
        lambda row: " ".join(
            row.fillna("").astype(str).tolist()
        ),
        axis=1
    )

    ein_ratio = (
        combined
        .str.contains(
            r"\b\d{2}-\d{7}\b",
            regex=True
        )
        .mean()
    )

    score += 0.25 * ein_ratio

    field_ratio = (
        sum(
            c in df.columns
            for c in VALUE_LABELS
        )
        / len(VALUE_LABELS)
    )

    score += 0.25 * field_ratio

    numeric_cols = [
        c
        for c in VALUE_LABELS
        if c in df.columns
    ]

    if numeric_cols:

        numeric_valid = (
            df[numeric_cols]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .notna()
            .mean()
            .mean()
        )

        score += 0.25 * numeric_valid

    # Require complete identity fields before reporting high confidence.
    required_identity = {"NAME", "P/S", "EIN"}
    if required_identity.issubset(df.columns):
        valid_identity = (
            df["NAME"].fillna("").astype(str).str.strip().ne("")
            & df["P/S"].fillna("").astype(str).str.fullmatch(r"[PS]")
            & df["EIN"].fillna("").astype(str).str.fullmatch(
                r"(?:[\d*]{2}-[\d*]{7}|APPLIED FOR)"
            )
        )
        if not valid_identity.all():
            score = min(score, 0.5)

    return round(
        min(score, 1.0),
        3
    )


def repair_table_numeric_cells_from_source(original_pdf, df):
    """Recover OCR-truncated numeric cells from ruled-table source images.

    This is intentionally conservative: a repair is accepted only when a
    source row has the same EIN and the same count of populated value cells.
    """
    required = {"EIN", *VALUE_LABELS}
    if df is None or df.empty or not required.issubset(df.columns):
        return df

    repaired = df.copy()
    ein_to_index = {
        re.sub(r"\s+", "", str(ein)): index
        for index, ein in repaired["EIN"].items()
    }
    changes = 0
    sparse_eins = []
    source = pymupdf.open(str(original_pdf))
    try:
        for page in source:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(300 / 72, 300 / 72),
                colorspace=pymupdf.csGRAY,
                alpha=False,
            )
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
            )
            text = pytesseract.image_to_string(
                remove_table_rules_for_ocr(image),
                config="--psm 6",
                timeout=60,
            )
            sparse_text = pytesseract.image_to_string(
                remove_table_rules_for_ocr(image),
                config="--psm 11",
                timeout=60,
            )
            sparse_eins.extend(
                re.findall(r"\b\d{2}\s*-\s*\d{7}\b", sparse_text)
            )
            for line in text.splitlines():
                match = re.search(r"\b(\d{2}\s*-\s*\d{7})\b", line)
                if not match:
                    continue
                ein = re.sub(r"\s+", "", match.group(1))
                row_index = ein_to_index.get(ein)
                if row_index is None:
                    continue
                tokens = re.findall(
                    r"\d+(?:\s*,\s*\d{3})?",
                    line[match.end():],
                )
                source_values = [
                    int(re.sub(r"[\s,]", "", token))
                    for token in tokens
                ]
                current = repaired.loc[row_index, VALUE_LABELS]
                populated = [
                    column
                    for column in VALUE_LABELS
                    if int(current[column]) != 0
                ]
                if len(populated) != len(source_values):
                    continue
                for column, value in zip(populated, source_values):
                    if int(repaired.at[row_index, column]) != value:
                        repaired.at[row_index, column] = value
                        changes += 1
    finally:
        source.close()

    # Use sparse OCR only for one-character EIN corrections.
    if len(sparse_eins) == len(repaired):
        for row_index, source_ein in zip(repaired.index, sparse_eins):
            source_ein = re.sub(r"\s+", "", source_ein)
            current_ein = str(repaired.at[row_index, "EIN"])
            if (
                len(source_ein) == len(current_ein)
                and sum(a != b for a, b in zip(source_ein, current_ein)) == 1
            ):
                repaired.at[row_index, "EIN"] = source_ein
                changes += 1

    if changes:
        print(
            f"---- REPAIRED {changes} OCR-TRUNCATED TABLE FIELD(S) ----"
        )
    return repaired



def pdf_attribute_pipeline(in_file):
    """Inspect a PDF and collect attributes that determine its extraction path."""

    type_attributes.clear()

    print(
        "---- DETERMINING FILE ATTRIBUTES ----"
    )

    with pdfplumber.open(in_file) as pdf:

        if not pdf.pages:
            raise ValueError(
                "PDF has no pages."
            )

        rotate_bool = to_rotate(
            pdf.pages[0]
        )

    if rotate_bool:

        print(
            "---- ROTATING FILE ----"
        )

        in_file = rotate(in_file)

    with pdfplumber.open(in_file) as pdf:

        raw_ssn = to_check_decode(
            pdf.pages[0]
        )

        decode_bool, _ = to_decode(
            raw_ssn
        )

        dedup_bool, char_df = to_dedup(
            pdf.pages[0]
        )

        if decode_bool:
            char_df = decode(char_df)

        if dedup_bool:

            all_chars = dedup(
                pdf.pages[0]
            )

            char_df = pd.DataFrame(
                all_chars
            )

        if "page_number" in char_df.columns:

            char_df["page"] = (
                char_df["page_number"] - 1
            )

        require_extractable_characters(
            char_df,
            "PDF attribute detection"
        )

        line_df, char_df = lines(
            char_df
        )

        if line_df.empty:
            raise ValueError(
                "PDF attribute detection produced no usable text lines."
            )

        line_df, _ = to_collapse(
            line_df
        )

        line_df, _, _ = header_delimiter(
            line_df
        )

    return (
        in_file,
        type_attributes.copy()
    )



def pdf_pipeline(
    in_file,
    in_type_attributes,
    visual_debugging=False,
    num_cols=7
):
    """Extract raw line and character data from a PDF using the selected preprocessing path."""

    print(
        "---- EXTRACTING DATA FROM PDF ----"
    )

    text_list = []
    x0_list = []
    x1_list = []
    top_list = []
    bottom_list = []
    page_list = []

    all_chars = []

    with pdfplumber.open(in_file) as pdf:

        for raw_page in pdf.pages:

            for char in raw_page.chars:

                text_list.append(
                    char["text"]
                )

                x0_list.append(
                    char["x0"]
                )

                x1_list.append(
                    char["x1"]
                )

                top_list.append(
                    char["top"]
                )

                bottom_list.append(
                    char["bottom"]
                )

                page_list.append(
                    char["page_number"] - 1
                )

    char_df = pd.DataFrame({
        "text": text_list,
        "x0": x0_list,
        "bottom": bottom_list,
        "x1": x1_list,
        "top": top_list,
        "page": page_list
    })

    require_extractable_characters(
        char_df,
        "PDF extraction"
    )


    if in_type_attributes.get(
        "duplicated",
        False
    ):

        print(
            "---- DEDUPLICATING DUPLICATE CHARACTERS ----"
        )

        with pdfplumber.open(in_file) as pdf:

            for page in pdf.pages:

                all_chars.extend(
                    dedup(page)
                )

        char_df = pd.DataFrame(
            all_chars
        )

        if "page_number" in char_df.columns:

            char_df["page"] = (
                char_df["page_number"] - 1
            )

        needed_cols = [
            "text",
            "x0",
            "bottom",
            "x1",
            "top",
            "page"
        ]

        char_df = char_df[
            needed_cols
        ]


    if in_type_attributes.get(
        "encoded",
        False
    ):

        print(
            "---- DECODING ENCODED CHARACTERS ----"
        )

        char_df = decode(
            char_df
        )


    full_line_df, char_df = lines(
        char_df
    )

    if full_line_df.empty:
        raise ValueError(
            "PDF extraction produced no usable text lines."
        )

    full_line_df = remove_totals(
        full_line_df
    )

    print(
        "---- REMOVING FOOTERS ----"
    )

    full_line_df, _ = to_collapse(
        full_line_df
    )

    full_line_df, _, _ = header_delimiter(
        full_line_df
    )

    print(
        "---- FINDING AND CLASSIFYING HEADERS ----"
    )

    header_line_df = full_line_df.copy()

    horizontal_lines = header_lines(
        char_df,
        header_line_df
    )


    if in_type_attributes.get(
        "two_line_entities",
        False
    ):

        full_line_df = classify_headers_two_line_entities(
            full_line_df
        )

    elif in_type_attributes.get(
        "delimiter",
        False
    ):

        full_line_df = (
            classify_headers_with_delimiters(
                full_line_df
            )
        )

    elif not in_type_attributes.get(
        "data_spans_rows",
        False
    ):

        full_line_df = (
            classify_headers_no_delimiters_collapsed(
                full_line_df
            )
        )

    else:

        full_line_df = (
            classify_headers_no_delimiters_uncollapsed(
                full_line_df
            )
        )

    full_line_df = remove_duplicate_headers(
        full_line_df
    )

    print(
        "---- REMOVING REPEATED HEADERS ----"
    )


    min_all = None
    max_ein = None

    if in_type_attributes.get(
        "two_line_entities",
        False
    ):

        print(
            "---- RECONSTITUTING TABLE FROM NAME / P-S-EIN LINE PAIRS ----"
        )

        clean_char_df, full_line_df = join_two_line_entity_chars(
            full_line_df,
            char_df
        )

    elif in_type_attributes.get(
        "data_spans_rows",
        False
    ):

        print(
            "---- RECONSTITUTING TABLE FROM SPLIT ROWS ----"
        )

        full_line_df = classify_noncollapsed_rows(
            full_line_df
        )

        full_line_df = collapse_rows(
            full_line_df
        )

        full_line_df, min_all, max_ein = shift_rows(
            full_line_df
        )

        print(
            "---- REORGANIZING TABLE WITH COLLAPSED ROWS ----"
        )

        clean_char_df, _ = (
            join_to_char_df_uncollapsed(
                full_line_df,
                char_df
            )
        )

    else:

        clean_char_df, _ = (
            join_to_char_df_normal(
                full_line_df,
                char_df
            )
        )


    word_df = words(
        clean_char_df
    )


    if in_type_attributes.get(
        "delimiter",
        False
    ):

        vertical_lines = (
            columns_header_delimiter(
                char_df,
                header_line_df
            )
        )

        if (
            in_type_attributes.get(
                "data_spans_rows",
                False
            )
            and min_all is not None
            and max_ein is not None
        ):

            vertical_lines.extend(
                [
                    min_all,
                    min_all - max_ein
                ]
            )

        vertical_lines = sorted(
            set(vertical_lines)
        )

    else:
        word_df, vertical_lines = columns_words_no_delimiter(
            clean_char_df,
            full_line_df,
            num_cols=None
        )

    print(
        "---- SCANNING FOR COLUMN AND ROW BREAKS ----"
    )


    word_df = assign_words_to_columns(
        word_df,
        vertical_lines
    )

    table_df = reconstitute_table(
        word_df,
        None
    )

    table_df = merge_wrapped_entities(
        table_df
    )

    clean_table_df = table_cleaning(
        table_df
    )

    print(
        "---- CLEANING TABLE ENTRIES ----"
    )


    debug_image = None

    if visual_debugging:

        print(
            "---- GENERATING DEBUG IMAGE ----"
        )

        with pdfplumber.open(in_file) as pdf:

            debug_image = (
                pdf.pages[0]
                .to_image(resolution=150)
            )

            page_width = pdf.pages[0].width
            page_height = pdf.pages[0].height

            for line in vertical_lines:

                if (
                    0 <= line <= page_width
                ):

                    debug_image.draw_line(
                        (
                            (line, 0),
                            (line, page_height)
                        ),
                        stroke="green",
                        stroke_width=2
                    )

            for line in horizontal_lines:

                if (
                    0 <= line <= page_height
                ):

                    debug_image.draw_line(
                        (
                            (0, line),
                            (page_width, line)
                        ),
                        stroke="red",
                        stroke_width=2
                    )

    print(
        clean_table_df.head(5)
    )

    return (
        clean_table_df,
        debug_image
    )



def process_schedule_e(
    original_pdf,
    output_folder=None,
    working_folder=None,
    visual_debugging=False,
    force_ocr=None,
    minimum_confidence=0.75
):
    """Extract one Schedule E PDF into a validated table and diagnostic artifacts."""

    print(
        "\n============================================"
    )

    print(
        "SCHEDULE E PROCESSING"
    )

    print(
        "============================================"
    )


    processed_pdf, ocr_diagnostics = (
        prepare_pdf_with_pymupdf4llm(
            original_pdf,
            working_folder=working_folder,
            force_ocr=force_ocr,
            ocr_language="eng",
            ocr_dpi=300
        )
    )


    processed_pdf, attributes = (
        pdf_attribute_pipeline(
            processed_pdf
        )
    )

    attributes["ocr"] = ocr_diagnostics

    print(
        "---- FILE ATTRIBUTES ----"
    )

    print(
        attributes
    )


    df, debug_image = pdf_pipeline(
        processed_pdf,
        attributes,
        visual_debugging=visual_debugging
    )

    df = repair_table_numeric_cells_from_source(
        original_pdf,
        df,
    )


    confidence = extraction_confidence(
        df
    )

    print(
        "---- EXTRACTION CONFIDENCE:",
        f"{confidence:.1%}",
        "----"
    )

    if confidence >= 0.90:

        review_status = (
            "HIGH CONFIDENCE"
        )

    elif confidence >= 0.75:

        review_status = (
            "REVIEW RECOMMENDED"
        )

    else:

        review_status = (
            "MANUAL REVIEW REQUIRED"
        )

    print(
        review_status
    )

    if confidence < minimum_confidence:
        raise ValueError(
            "Extraction confidence is below the output threshold "
            f"({confidence:.1%} < {minimum_confidence:.1%}). "
            "No CSV was written because the extracted table is not reliable."
        )


    original_path = Path(
        original_pdf
    )

    if output_folder is None:

        output_dir = (
            original_path.parent
        )

    else:

        output_dir = Path(
            output_folder
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir
        /
        (
            original_path.stem
            + ".csv"
        )
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        "---- OUTPUTTING CSV FILE ----"
    )

    print(
        csv_path
    )

    return {
        "dataframe": df,
        "csv_path": str(csv_path),
        "processed_pdf": processed_pdf,
        "confidence": confidence,
        "review_status": review_status,
        "attributes": attributes,
        "debug_image": debug_image
    }


def process_batch(input_folder, output_folder, working_folder, force_ocr=None):
    """Process every PDF under an input folder and record outputs or failures."""
    """Process each source PDF in ``input_folder`` into ``output_folder``."""
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    work_dir = Path(working_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "failed_files.txt"
    failure_log.write_text("", encoding="utf-8")

    # Preserve input subfolders to prevent filename collisions.
    for file_path in sorted(input_dir.rglob("*.pdf")):
        filename = file_path.name
        name = filename.lower()
        if any(marker in name for marker in ("_processed", "_raster", "portrait")):
            continue

        relative_parent = file_path.relative_to(input_dir).parent
        file_output_dir = output_dir / relative_parent
        file_work_dir = work_dir / relative_parent

        print(f"\n{'#' * 48}\nPROCESSING: {filename}\n{'#' * 48}")
        try:
            result = process_schedule_e(
                original_pdf=str(file_path),
                output_folder=file_output_dir,
                working_folder=file_work_dir,
                visual_debugging=False,
                force_ocr=force_ocr,
                minimum_confidence=0.75,
            )
            print(f"SUCCESS: {result['csv_path']}")
            print(f"CONFIDENCE: {result['confidence']:.1%}")
            print(f"STATUS: {result['review_status']}")
        except Exception as exc:
            print(f"FAILED: {filename}\n{type(exc).__name__}: {exc}")
            relative_path = file_path.relative_to(input_dir)
            with failure_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{relative_path}\t{type(exc).__name__}: {exc}\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Schedule E Part II data from PDFs."
    )
    parser.add_argument("input_folder", help="Folder containing source PDF files")
    parser.add_argument("output_folder", help="Folder for generated CSV files")
    parser.add_argument("working_folder", help="Folder for temporary OCR files")
    parser.add_argument("--force-ocr", action="store_true")
    args = parser.parse_args()
    process_batch(
        args.input_folder,
        args.output_folder,
        args.working_folder,
        force_ocr=True if args.force_ocr else None,
    )
