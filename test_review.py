"""Tests for the deterministic layer.

Everything here is a pure function over data: no network, no model, no monday.
That is the half of the system that must never change its answer, so it is the
half worth pinning down.

Run: venv/bin/python -m pytest test_review.py -q
"""
import datetime as dt

import pandas as pd
import pytest

import board_review as br


# ---------------------------------------------------------------- helpers

def col(cid, title, ctype, settings=None, description=None):
    return {"id": cid, "title": title, "type": ctype,
            "description": description, "settings_str": settings}


def board(columns, rows):
    """rows: list of dicts keyed by column id."""
    meta = {"name": "B", "description": None, "columns": columns}
    items = [{"id": str(i), "name": r.pop("__name__", f"item{i}"),
              "column_values": [{"id": k, "type": "text", "text": v, "value": None}
                                for k, v in r.items()]}
             for i, r in enumerate(rows)]
    return meta, items


def profile(values, monday_type="text", name="c", labels=None):
    return br.profile_column(pd.Series(values), name, monday_type, labels or [])


# ---------------------------------------------------------------- hygiene

def test_blank_strings_become_null_before_typing():
    """One empty string in a numeric column used to force the whole column to
    text, which is why nothing could be summed."""
    p = profile(["100", "200", "", "300"], "numbers")
    assert p.storage_type == "number"
    assert p.nulls == 1


def test_whitespace_only_counts_as_blank():
    assert br.is_blank("   ") and br.is_blank("") and br.is_blank(None)
    assert not br.is_blank("0")


# ---------------------------------------------------------------- typing

def test_type_is_measured_not_declared():
    """monday calling a column text does not make it text."""
    dates = profile(["2025-01-01", "2025-06-15", "2026-02-02"], "text")
    nums = profile(["1000", "2500", "310"], "text")
    assert dates.storage_type == "date"
    assert nums.storage_type == "number"


def test_bare_digits_are_not_dates():
    """pandas will read '1000' as a year given the chance."""
    p = profile(["1000", "1037", "1074", "2011"], "text")
    assert p.storage_type == "number"


def test_month_names_are_not_dates():
    p = profile(["Dec", "June", "November", "May", "Dec"], "status")
    assert p.semantic_type == "month"
    assert p.storage_type != "date"


def test_column_below_threshold_stays_text_and_is_flagged():
    vals = ["TBD", "approx 5L", "n/a"] + [str(i * 1000) for i in range(7)]
    p = profile(vals, "numbers")
    assert p.storage_type == "text"
    assert p.cast_expr and "TRY_CAST" in p.cast_expr
    assert any("numeric" in w for w in p.warnings)


def test_quantity_with_units_is_split_not_summed():
    p = profile(["5360 HA", "4", "59.33", "600"], "dropdown")
    assert p.semantic_type == "quantity"
    assert br.split_quantity("5360 HA") == (5360.0, "HA")
    assert br.split_quantity("59.33") == (59.33, None)


def test_identifier_is_not_treated_as_a_category():
    """A dropdown with a distinct value per row is an id, not a category."""
    p = profile([f"SDPLDEAL-{i:03d}" for i in range(40)], "dropdown", name="Serial #")
    assert p.semantic_type == "identifier"


# ---------------------------------------------------------------- quantities

def test_zero_is_reported_separately_from_null():
    p = profile(["100", "0", "0", "250", ""], "numbers")
    assert p.zeros == 2 and p.nulls == 1 and p.real == 2


def test_severity_follows_proportion():
    assert br.severity_for(1.0) == "Critical"
    assert br.severity_for(0.6) == "High"
    assert br.severity_for(0.3) == "Medium"
    assert br.severity_for(0.05) == "Low"


# ---------------------------------------------------------------- verdicts

@pytest.mark.parametrize("state,kind", [
    ("Won", "completed"), ("Dead", "abandoned"), ("Open", "active"),
    ("Pause / struck", "abandoned"), ("Not Started", "active"),
])
def test_state_classification(state, kind):
    assert br.classify_state(state) == kind


def test_gap_on_a_finished_record_is_a_defect():
    assert br.verdict_for("completed", 0.61) == "DEFECT"


def test_gap_on_an_abandoned_record_is_expected():
    assert br.verdict_for("abandoned", 0.94) == "expected"


def test_forecast_field_may_vanish_once_a_record_closes():
    """A closure probability on a won deal is meaningless, not missing."""
    assert br.verdict_for("completed", 0.82, forward_looking=True) == "expected"


# ---------------------------------------------------------------- identity

def test_common_stem_is_stable_regardless_of_iteration_order():
    """This flipped between processes and silently disabled the guardrail
    against inventing a join key."""
    vals = {"ACME_01", "BETACORP_02", "ACME_03", "BETACORP_04"}
    assert br.common_stem(vals) == br.common_stem(set(reversed(sorted(vals))))
    assert br.common_stem({"WOCOMPANY_001", "WOCOMPANY_052"}) == "wocompany"


def test_sql_identifiers_are_escaped():
    assert br.quote_ident('Serial #') == '"Serial #"'
    assert br.quote_ident('He said "hi"') == '"He said ""hi"""'


# ---------------------------------------------------------------- joins

def _two_boards(left_vals, right_vals, left_name="Ref", right_name="Ref"):
    # Item names repeat deliberately: build_raw_frame always adds monday's
    # "name" column, and a unique one there would win as a safer candidate —
    # which is correct behaviour, but not what these tests are exercising.
    ca = [col("a1", left_name, "text")]
    cb = [col("b1", right_name, "text")]
    ma, ia = board(ca, [{"a1": v, "__name__": "shared"} for v in left_vals])
    mb, ib = board(cb, [{"b1": v, "__name__": "shared"} for v in right_vals])
    da = br.build_raw_frame(ia, ma); dbf = br.build_raw_frame(ib, mb)
    pa = br.review_columns(da, ma); pb = br.review_columns(dbf, mb)
    return br.compare_boards(pa, da, "A", pb, dbf, "B", "ta", "tb")


def test_repeating_key_is_reported_as_unsafe_with_its_fan_out():
    left = [f"P{i%12}" for i in range(48)]     # 12 labels, each 4x
    right = [f"P{i%12}" for i in range(36)]    # 12 labels, each 3x
    _findings, join = _two_boards(left, right)
    assert join["kind"] == "many-to-many"
    assert join["fanout"] > 1


def test_unique_key_on_both_sides_is_safe():
    vals = [f"ID{i:03d}" for i in range(40)]
    _findings, join = _two_boards(vals, vals)
    assert join["kind"] == "one-to-one"


def test_a_safe_join_outranks_a_wider_unsafe_one():
    """Coverage alone must not win: a key that identifies rows beats one that
    merely matches more values."""
    ca = [col("a1", "Ref", "text"), col("a2", "Grp", "text")]
    cb = [col("b1", "Ref", "text"), col("b2", "Grp", "text")]
    ma, ia = board(ca, [{"a1": f"ID{i:03d}", "a2": f"G{i%12}", "__name__": "s"}
                        for i in range(40)])
    mb, ib = board(cb, [{"b1": f"ID{i:03d}", "b2": f"G{i%12}", "__name__": "s"}
                        for i in range(40)])
    da, dbf = br.build_raw_frame(ia, ma), br.build_raw_frame(ib, mb)
    _f, join = br.compare_boards(br.review_columns(da, ma), da, "A",
                                 br.review_columns(dbf, mb), dbf, "B", "ta", "tb")
    assert join["kind"] == "one-to-one" and join["left"] == "Ref"


def test_join_names_its_own_tables():
    vals = [f"ID{i:03d}" for i in range(40)]
    _f, join = _two_boards(vals, vals)
    assert join["left_table"] == "ta" and join["right_table"] == "tb"


# ---------------------------------------------------------------- structure

def test_structure_hash_ignores_data_and_tracks_schema():
    cols = [col("c1", "A", "text")]
    m1, _ = board(cols, [{"c1": "x"}])
    m2, _ = board(cols, [{"c1": "totally different"} for _ in range(50)])
    assert br.structure_key("1", m1) == br.structure_key("1", m2)
    m3, _ = board(cols + [col("c2", "B", "text")], [{"c1": "x", "c2": "y"}])
    assert br.structure_key("1", m1) != br.structure_key("1", m3)


def test_duplicate_titles_do_not_overwrite_each_other():
    cols = [col("c1", "Date", "text"), col("c2", "Date", "text")]
    meta, items = board(cols, [{"c1": "a", "c2": "b"}])
    df = br.build_raw_frame(items, meta)
    assert "Date" in df.columns and "Date_1" in df.columns


def test_mondays_own_name_column_is_not_duplicated():
    """It always comes back empty and produced a phantom Critical finding."""
    cols = [col("nm", "Name", "name"), col("c1", "Other", "text")]
    meta, items = board(cols, [{"nm": "", "c1": "v"}])
    df = br.build_raw_frame(items, meta)
    assert "Name" not in df.columns
    assert "name" in df.columns


# ---------------------------------------------------------------- resilience

@pytest.mark.parametrize("rows,cols", [
    (0, 2), (1, 1), (5, 0),
])
def test_degenerate_boards_do_not_raise(rows, cols):
    columns = [col(f"c{i}", f"C{i}", "text") for i in range(cols)]
    meta, items = board(columns, [{f"c{i}": "v" for i in range(cols)}
                                  for _ in range(rows)])
    review, df = br.review_board("b", meta, items, "t", label="T")
    br.build_database({"t": df}, review.findings)


def test_unparseable_values_are_kept_not_dropped():
    p = profile(["1", "2", "not a number", "4"], "numbers")
    assert p.rows == 4 and p.nulls == 0
