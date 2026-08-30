"""Board review: measure, type, check, and describe a monday board.

Runs once per board structure. Everything it produces is derived from the data
and the board's own metadata, so it works on boards it has never seen and on
boards with no column descriptions.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

MONEY_CUES = ("amount", "rupee", "value", "price", "cost", "revenue", "billed",
              "collected", "receivable", "invoice", "budget", "fee")
QTY_CUES = ("quantit", "qty", "count", "volume", "area", "hours", "units")
PCT_CUES = ("percent", "%", "probability", "rate", "margin")
ID_CUES = ("id", "code", "serial", "no.", "number", "#", "ref")

CONVERT_THRESHOLD = 0.95
HIGH_ZERO_SHARE = 0.20


def quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping any embedded double quote."""
    return '"' + str(name).replace('"', '""') + '"'


def common_stem(values) -> str:
    """The alphabetic prefix shared by most values: {"WOCOMPANY_002", ...} ->
    "wocompany".

    Sorted before counting so ties resolve the same way every run. Taking an
    arbitrary set element made this flip between processes on identical data.
    """
    from collections import Counter
    stems = Counter(re.match(r"[A-Za-z]*", str(v)).group(0).lower()
                    for v in sorted(str(x) for x in values))
    if not stems:
        return ""
    stem = min((s for s, n in stems.items()
                if n == max(stems.values())), default="")
    return stem if stems[stem] >= max(1, len(values) * 0.5) else ""


def is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and not v.strip()


# ---------------------------------------------------------------- structure key

def structure_key(board_id: str, metadata: Dict[str, Any]) -> str:
    """Hash of the board's shape. Data changes do not move it; columns do."""
    parts = []
    for c in sorted(metadata.get("columns", []), key=lambda c: c["id"]):
        parts.append(f"{c['id']}|{c['title']}|{c['type']}|{c.get('settings_str') or ''}")
    blob = f"{board_id}::" + "||".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- raw assembly

def build_raw_frame(items: List[Dict[str, Any]], metadata: Dict[str, Any]) -> pd.DataFrame:
    """Flatten items to a frame of raw values, with blanks normalised to None.

    Column titles are made unique; duplicates get a numeric suffix so no column
    silently overwrites another.
    """
    titles: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for col in metadata.get("columns", []):
        if col["type"] == "name":
            continue  # monday's own name column duplicates item.name
        t = col["title"]
        if t in seen:
            seen[t] += 1
            titles[col["id"]] = f"{t}_{seen[t]}"
        else:
            seen[t] = 0
            titles[col["id"]] = t

    rows = []
    for item in items:
        row: Dict[str, Any] = {"item_id": item["id"], "name": item["name"]}
        for cv in item.get("column_values", []):
            if cv.get("type") == "name":
                continue  # duplicates item.name, and always empty here
            val = cv.get("text")
            if is_blank(val) and cv.get("value"):
                try:
                    parsed = json.loads(cv["value"])
                    if isinstance(parsed, dict) and "date" in parsed:
                        val = parsed["date"]
                except Exception:
                    pass
            row[titles.get(cv["id"], cv["id"])] = None if is_blank(val) else val
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in titles.values():
        if col not in df.columns:
            df[col] = None
    return df


def monday_types(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Declared monday type per (uniquified) column title."""
    out: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for col in metadata.get("columns", []):
        if col["type"] == "name":
            continue
        t = col["title"]
        if t in seen:
            seen[t] += 1
            t = f"{t}_{seen[t]}"
        else:
            seen[t] = 0
        out[t] = col["type"]
    return out


def label_sets(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Declared allowed values for status/dropdown columns."""
    out: Dict[str, List[str]] = {}
    seen: Dict[str, int] = {}
    for col in metadata.get("columns", []):
        if col["type"] == "name":
            continue
        t = col["title"]
        if t in seen:
            seen[t] += 1
            t = f"{t}_{seen[t]}"
        else:
            seen[t] = 0
        if col["type"] not in ("status", "dropdown", "color"):
            continue
        try:
            labels = json.loads(col.get("settings_str") or "{}").get("labels", {})
        except Exception:
            continue
        vals = list(labels.values()) if isinstance(labels, dict) else [
            l.get("name") for l in labels if isinstance(l, dict)
        ]
        out[t] = [v for v in vals if v is not None]
    return out


def descriptions_from_monday(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Author-written descriptions, if the board has any. Quotes stripped."""
    out: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for col in metadata.get("columns", []):
        if col["type"] == "name":
            continue
        t = col["title"]
        if t in seen:
            seen[t] += 1
            t = f"{t}_{seen[t]}"
        else:
            seen[t] = 0
        d = (col.get("description") or "").strip().strip('"').strip()
        if d:
            out[t] = d
    return out


# ---------------------------------------------------------------- profiling

@dataclass
class ColumnProfile:
    name: str
    monday_type: str
    rows: int = 0
    nulls: int = 0
    zeros: int = 0
    distinct: int = 0
    samples: List[str] = field(default_factory=list)
    top_values: List[tuple] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    convert_rate: float = 0.0
    storage_type: str = "text"
    semantic_type: str = "text"
    cast_expr: Optional[str] = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    description: str = ""
    description_source: str = "generated"
    semantic_source: str = "measured"
    inferred_meaning: str = ""
    authored_meaning: str = ""
    inferred_caution: str = ""
    role: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def null_pct(self) -> float:
        return self.nulls / self.rows if self.rows else 0.0

    @property
    def zero_pct(self) -> float:
        return self.zeros / self.rows if self.rows else 0.0

    @property
    def filled(self) -> int:
        return self.rows - self.nulls

    @property
    def real(self) -> int:
        """Values that are neither null nor a zero standing in for absence."""
        return self.rows - self.nulls - self.zeros


def _clean_number(v: Any) -> Optional[float]:
    try:
        s = str(v)
        s = re.sub(r"[₹$€£,\s]", "", s)
        return float(s)
    except (ValueError, TypeError):
        return None


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_LOOKUP = {}
for _i, _m in enumerate(MONTH_NAMES, start=1):
    MONTH_LOOKUP[_m.lower()] = (_m, _i)
    MONTH_LOOKUP[_m[:3].lower()] = (_m, _i)
MONTH_LOOKUP["sept"] = ("September", 9)

QTY_PATTERN = re.compile(r"^\s*([0-9][0-9,]*\.?[0-9]*)\s*([A-Za-z%²/ ]{0,12})\s*$")


def canonical_month(value: Any) -> Optional[tuple]:
    """('November', 11) for any of 'Nov', 'nov', 'November'."""
    if is_blank(value):
        return None
    return MONTH_LOOKUP.get(str(value).strip().lower())


def looks_like_months(values: List[Any], threshold: float = 0.8) -> bool:
    non_null = [v for v in values if not is_blank(v)]
    if len(non_null) < 3:
        return False
    hits = sum(1 for v in non_null if canonical_month(v))
    return hits / len(non_null) >= threshold


def split_quantity(value: Any) -> tuple:
    """('5360 HA') -> (5360.0, 'HA');  ('59.33') -> (59.33, None)."""
    if is_blank(value):
        return None, None
    match = QTY_PATTERN.match(str(value))
    if not match:
        return None, None
    number = _clean_number(match.group(1))
    unit = (match.group(2) or "").strip() or None
    return number, unit


def _semantic_type(name: str, monday_type: str, storage: str,
                   distinct: int, rows: int) -> str:
    low = name.lower()
    if storage == "date":
        return "date"
    if monday_type in ("status", "dropdown", "color"):
        if rows and (distinct / rows > 0.5 or distinct > 40):
            return "identifier"
        return "categorical"
    if storage == "number":
        if any(c in low for c in PCT_CUES):
            return "percentage"
        # Quantity is checked first: "Quantity billed" contains a money cue
        # ("billed") but is a count, and the more specific cue must win.
        if any(c in low for c in QTY_CUES):
            return "quantity"
        if any(c in low for c in MONEY_CUES):
            return "money"
        return "number"
    if any(c in low for c in ID_CUES) or (rows and distinct / rows > 0.9):
        return "identifier"
    if rows and distinct and distinct / rows < 0.2 and distinct < 50:
        return "categorical"
    return "text"


def profile_column(series: pd.Series, name: str, monday_type: str,
                   labels: List[str]) -> ColumnProfile:
    values = list(series)
    rows = len(values)
    non_null = [v for v in values if not is_blank(v)]

    p = ColumnProfile(name=name, monday_type=monday_type, rows=rows,
                      nulls=rows - len(non_null), labels=labels)
    p.distinct = len({str(v) for v in non_null})
    p.samples = [str(v) for v in non_null[:5]]
    # The values are already in memory, so the distribution is nearly free —
    # and it saves the agent a round-trip to discover it.
    if non_null and p.distinct <= 60:
        from collections import Counter
        p.top_values = Counter(str(v) for v in non_null).most_common(8)

    # Storage type. monday's declared type is only a hint — a column it calls
    # text may hold dates, and one it calls a dropdown may hold numbers. So every
    # column is *tried* as a number and as a date, and whatever actually parses
    # wins. Nothing here is inferred; it is measured.
    def _try(parser):
        if not non_null:
            return 0.0, []
        out = [parser(v) for v in non_null]
        ok = [v for v in out if v is not None]
        return len(ok) / len(non_null), ok

    def _try_dates():
        """Vectorised, and only on values that could be dates.

        Parsing scalars one at a time costs ~40ms per thousand values, which
        across every column of every board dominated the whole review.
        """
        if not non_null:
            return 0.0, []
        candidates, keep = [], []
        for v in non_null:
            s = str(v).strip()
            # A bare month name is not a date; bare digits are not either.
            if canonical_month(s) or (
                    not re.search(r"[-/.]", s) and not re.search(r"[A-Za-z]{3}", s)):
                continue
            candidates.append(s)
        if not candidates:
            return 0.0, []
        try:
            parsed = pd.to_datetime(pd.Series(candidates), errors="coerce",
                                    format="mixed")
        except Exception:
            return 0.0, []
        keep = [d for d in parsed if not pd.isna(d)]
        return len(keep) / len(non_null), keep

    low_name = name.lower()
    # Identifiers made of digits must not become numbers: a code is not a value.
    id_like = any(c in low_name for c in ID_CUES)

    num_rate, nums = (0.0, []) if id_like else _try(_clean_number)
    date_rate, parsed_dates = _try_dates()

    if num_rate >= CONVERT_THRESHOLD and num_rate >= date_rate:
        p.storage_type, p.convert_rate = "number", num_rate
        p.zeros = sum(1 for v in nums if v == 0)
    elif date_rate >= CONVERT_THRESHOLD:
        p.storage_type, p.convert_rate = "date", date_rate
        if parsed_dates:
            p.date_min = str(min(parsed_dates).date())
            p.date_max = str(max(parsed_dates).date())
    else:
        p.convert_rate = max(num_rate, date_rate) if non_null else 0.0
        # Only complain when monday promised a type the data doesn't honour.
        if monday_type in ("numbers", "numeric") and non_null:
            p.warnings.append(
                f"only {num_rate:.0%} of values are numeric; kept as text and not "
                "usable for arithmetic")
            p.cast_expr = f"TRY_CAST(REPLACE({quote_ident(name)}, ',', '') AS DOUBLE)"
        elif monday_type == "date" and non_null:
            p.warnings.append(
                f"only {date_rate:.0%} of values parse as dates; kept as text")
            p.cast_expr = f"TRY_CAST({quote_ident(name)} AS DATE)"
        elif non_null and max(num_rate, date_rate) >= 0.5:
            kind = "numeric" if num_rate > date_rate else "date"
            p.warnings.append(
                f"{max(num_rate, date_rate):.0%} of values look {kind} but the rest "
                "do not; kept as text")

    p.semantic_type = _semantic_type(name, monday_type, p.storage_type,
                                     p.distinct, rows)

    # A dropdown of "5360 HA", "4", "59.33" is a quantity monday happens to
    # declare as a category. Detect it from the values, not the name.
    if p.storage_type == "text" and non_null and p.semantic_type in (
            "categorical", "text", "identifier", "number"):
        parsed = [split_quantity(v) for v in non_null]
        if sum(1 for n, _u in parsed if n is not None) / len(non_null) >= 0.8:
            p.semantic_type = "quantity"

    # Month-name columns: 'Dec' and 'November' in one column, unsortable as text.
    if p.storage_type == "text" and looks_like_months(non_null):
        p.semantic_type = "month"
        forms = {str(v).strip() for v in non_null}
        if any(len(f) <= 4 for f in forms) and any(len(f) > 4 for f in forms):
            p.warnings.append(
                "month names mix abbreviated and full forms; normalised to full "
                "names, with a companion ordinal column for chronological sorting")

    # Quantity columns holding a number and a unit in one string.
    if p.semantic_type == "quantity" and p.storage_type != "number" and non_null:
        parsed = [split_quantity(v) for v in non_null]
        units = {u for _n, u in parsed if u}
        if any(n is not None for n, _u in parsed):
            p.warnings.append(
                f"values combine a number with a unit ({', '.join(sorted(units)) or 'unitless'}); "
                "split into companion value and unit columns — sum within a unit, never across")
    return p


def apply_types(df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> pd.DataFrame:
    """Convert whole columns after assembly, so one blank cannot force a
    numeric column to text."""
    out = df.copy()
    for name, p in profiles.items():
        if name not in out.columns:
            continue
        if p.storage_type == "number":
            out[name] = pd.to_numeric(
                out[name].map(lambda v: None if is_blank(v) else _clean_number(v)),
                errors="coerce")
        elif p.storage_type == "date":
            out[name] = pd.to_datetime(
                out[name].map(lambda v: None if is_blank(v) else v),
                errors="coerce", format="mixed").dt.date
        elif p.semantic_type == "month":
            canon = out[name].map(canonical_month)
            out[name] = canon.map(lambda m: m[0] if m else None)
            out[f"{name} [month no]"] = canon.map(lambda m: m[1] if m else None)
        elif p.semantic_type == "quantity":
            parts = out[name].map(split_quantity)
            out[f"{name} [value]"] = parts.map(lambda x: x[0])
            out[f"{name} [unit]"] = parts.map(lambda x: x[1])
            out[name] = out[name].map(lambda v: None if is_blank(v) else str(v))
        else:
            out[name] = out[name].map(lambda v: None if is_blank(v) else str(v))
    return out


def review_columns(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, ColumnProfile]:
    types = monday_types(metadata)
    labels = label_sets(metadata)
    profiles: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        if col == "item_id":
            continue
        profiles[col] = profile_column(df[col], col,
                                       types.get(col, "text"),
                                       labels.get(col, []))
    return profiles


# ---------------------------------------------------------------- quality

def severity_for(pct: float) -> str:
    if pct >= 0.95:
        return "Critical"
    if pct >= 0.50:
        return "High"
    if pct >= 0.20:
        return "Medium"
    return "Low"


def _finding(board, column, issue, description, severity, pct=None):
    return {"board": board, "column": column, "issue_type": issue,
            "description": description, "severity": severity,
            "proportion": round(pct, 4) if pct is not None else None}


def find_status_column(profiles: Dict[str, ColumnProfile]) -> Optional[str]:
    """The column to condition null analysis on: a small, well-filled category."""
    cands = [p for p in profiles.values()
             if p.semantic_type == "categorical" and 2 <= p.distinct <= 12
             and p.null_pct < 0.5]
    if not cands:
        return None
    named = [p for p in cands if "status" in p.name.lower()]
    pool = named or cands
    return min(pool, key=lambda p: (p.distinct, p.null_pct)).name


def check_board(df: pd.DataFrame, profiles: Dict[str, ColumnProfile],
                board: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    status_col = find_status_column(profiles)

    for name, p in profiles.items():
        if p.rows == 0:
            continue

        if p.nulls == p.rows:
            findings.append(_finding(board, name, "Empty Column",
                f"Column holds no values at all ({p.rows} rows).",
                "Critical", 1.0))
        elif p.nulls:
            findings.append(_finding(board, name, "Missing Values",
                f"{p.nulls} of {p.rows} rows ({p.null_pct:.0%}) have no value.",
                severity_for(p.null_pct), p.null_pct))

        if p.storage_type == "number" and p.zero_pct >= HIGH_ZERO_SHARE:
            findings.append(_finding(board, name, "Zero As Missing",
                f"{p.zeros} of {p.rows} rows ({p.zero_pct:.0%}) are zero. "
                f"Only {p.real} rows carry a real figure. Zero is likely "
                "standing in for absence here.",
                severity_for(p.zero_pct), p.zero_pct))

        if p.warnings:
            for w in p.warnings:
                findings.append(_finding(board, name, "Type Inconsistency",
                    w, "High", 1 - p.convert_rate))

        for label in p.labels:
            if label and label.strip().lower() == name.split("_")[0].strip().lower():
                findings.append(_finding(board, name, "Header Row Imported",
                    f"'{label}' appears as an allowed value of its own column — "
                    "a spreadsheet header row imported as data. Exclude it.",
                    "High"))

        # Conditional nulls: is this gap expected, or a defect?
        if status_col and name != status_col and 0.05 < p.null_pct < 1.0:
            rates = []
            for state, grp in df.groupby(df[status_col].fillna("(blank)")):
                if len(grp) < 5:
                    continue
                miss = int(grp[name].isna().sum())
                rates.append((str(state), len(grp), miss, miss / len(grp)))
            if len(rates) >= 2:
                spread = max(r[3] for r in rates) - min(r[3] for r in rates)
                if spread >= 0.35:
                    forward = any(c in name.lower() for c in FORECAST_CUES)
                    judged = [(s, n, m, r, classify_state(s),
                               verdict_for(classify_state(s), r, forward))
                              for s, n, m, r in sorted(rates, key=lambda x: -x[3])]
                    defects = [j for j in judged if j[5] == "DEFECT"]
                    expected = [j for j in judged if j[5] == "expected"]

                    parts = []
                    if defects:
                        parts.append("DEFECT — " + "; ".join(
                            f"{s}: {m} of {n} {s.lower()} records ({r:.0%}) have no value"
                            for s, n, m, r, _k, _v in defects))
                    if expected:
                        why = ("this is a forecast field, superseded once a record closes"
                               if forward else "nothing to record on those")
                        parts.append("expected on " + ", ".join(
                            f"{s} ({r:.0%})" for s, _n, _m, r, _k, _v in expected)
                            + f" — {why}")
                    unclear = [j for j in judged if j[5] == "unclear"]
                    if unclear:
                        parts.append("unclassified: " + ", ".join(
                            f"{s} ({r:.0%})" for s, _n, _m, r, _k, _v in unclear))

                    worst = max((j[3] for j in defects), default=0.0)
                    findings.append(_finding(
                        board, name, "Conditional Null",
                        f"Varies by {status_col}. " + ". ".join(parts) + ".",
                        severity_for(worst) if defects else "Low",
                        worst if defects else spread))

    return findings


def find_twin_encodings(profiles: Dict[str, ColumnProfile], board: str) -> List[Dict[str, Any]]:
    """Pairs where one column marks absence with a blank and its twin with 0."""
    out = []
    nums = [p for p in profiles.values() if p.storage_type == "number"]
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            if a.real != b.real or a.real == 0:
                continue
            if (a.nulls and not a.zeros and b.zeros and not b.nulls) or \
               (b.nulls and not b.zeros and a.zeros and not a.nulls):
                out.append(_finding(board, f"{a.name} / {b.name}", "Twin Encoding",
                    f"Both cover the same {a.real} rows. One marks absence with a "
                    "blank, the other with zero — the same gap, encoded twice.",
                    "Medium"))
    return out


def compare_boards(pa: Dict[str, ColumnProfile], da: pd.DataFrame, na: str,
                   pb: Dict[str, ColumnProfile], db: pd.DataFrame, nb: str,
                   ta: str = "", tb: str = ""
                   ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Cross-board checks: shared vocabularies, and the best join key."""
    findings: List[Dict[str, Any]] = []
    shared_dims: List[Dict[str, Any]] = []
    disjoint: List[Dict[str, Any]] = []

    # Two identifier columns that look alike but share no values are separate
    # coding schemes. Saying so explicitly is what stops an agent inventing a
    # mapping between them.
    ident_a = {n: set(str(v) for v in da[n].dropna()) for n, pr in pa.items()
               if pr.semantic_type == "identifier"}
    ident_b = {n: set(str(v) for v in db[n].dropna()) for n, pr in pb.items()
               if pr.semantic_type == "identifier"}
    def confusable(x: str, xv, y: str, yv) -> float:
        """How likely is someone to mistake these two columns for each other?"""
        drop = {"code", "no", "id", "ref", "number"}
        wx = set(re.findall(r"[a-z]+", x.lower())) - drop
        wy = set(re.findall(r"[a-z]+", y.lower())) - drop
        word = len(wx & wy) / max(len(wx | wy), 1)
        sx, sy = common_stem(xv), common_stem(yv)
        # One scheme containing another's stem is how a prefix-stripping "match"
        # gets invented in the first place.
        shape = 1.0 if len(sx) >= 4 and len(sy) >= 4 and (sx in sy or sy in sx) else 0.0
        return max(word, shape)

    for an, av in ident_a.items():
        for bn, bv in ident_b.items():
            if len(av) < 10 or len(bv) < 10 or (av & bv):
                continue
            score = confusable(an, av, bn, bv)
            if score == 0:
                continue  # nobody would mistake these for each other
            disjoint.append({"left": an, "right": bn, "score": score,
                             "left_table": ta or na, "right_table": tb or nb,
                             "left_n": len(av), "right_n": len(bv)})
    disjoint.sort(key=lambda d: -d["score"])

    cat_a = {n: set(str(v) for v in da[n].dropna()) for n, p in pa.items()
             if p.semantic_type == "categorical"}
    cat_b = {n: set(str(v) for v in db[n].dropna()) for n, p in pb.items()
             if p.semantic_type == "categorical"}

    for na_col, va in cat_a.items():
        for nb_col, vb in cat_b.items():
            if not va or not vb:
                continue
            overlap = len(va & vb)
            if overlap >= 3:
                shared_dims.append({"left": na_col, "right": nb_col,
                                    "shared": overlap,
                                    "left_table": ta or na, "right_table": tb or nb})
            if overlap >= 3 and (va - vb or vb - va):
                only_b = sorted(vb - va)[:6]
                only_a = sorted(va - vb)[:6]
                findings.append(_finding(
                    f"{na} / {nb}", f"{na_col} / {nb_col}", "Vocabulary Split",
                    f"Same concept, different value sets: {len(va)} vs {len(vb)} values, "
                    f"{overlap} shared. Only in {nb}: {only_b or 'none'}. "
                    f"Only in {na}: {only_a or 'none'}. Comparisons across boards "
                    "cover different ground.", "Medium"))

    # A join key identifies records. Two things disqualify a candidate:
    # a category matches trivially (every board shares "Mining"), and a label
    # that repeats on either side turns a join into a cross product. Overlap
    # alone is not enough — uniqueness decides whether the join is safe.
    def joinable(profiles, col):
        p = profiles.get(col)
        if p is None:
            return col == "name"
        return p.semantic_type not in ("categorical", "percentage", "money", "date") \
            and p.distinct >= 10

    best = None
    for na_col in da.columns:
        if not joinable(pa, na_col):
            continue
        left_vals = [str(v) for v in da[na_col].dropna() if str(v).strip()]
        va = set(left_vals)
        if len(va) < 10:
            continue
        for nb_col in db.columns:
            if not joinable(pb, nb_col):
                continue
            right_vals = [str(v) for v in db[nb_col].dropna() if str(v).strip()]
            vb = set(right_vals)
            if len(vb) < 10:
                continue
            shared = len(va & vb)
            cov = shared / len(va)
            if shared < 10 or cov < 0.3:
                continue

            left_unique = len(va) == len(left_vals)
            right_unique = len(vb) == len(right_vals)
            if left_unique and right_unique:
                kind, rank = "one-to-one", 3
            elif left_unique or right_unique:
                kind, rank = "one-to-many", 2
            else:
                kind, rank = "many-to-many", 1

            # Row count an inner join would actually produce.
            from collections import Counter
            lc, rc = Counter(left_vals), Counter(right_vals)
            joined = sum(lc[k] * rc[k] for k in va & vb)

            cand = {"left": na_col, "right": nb_col,
                    "left_table": ta or na, "right_table": tb or nb,
                    "matched": shared,
                    "left_total": len(va), "coverage": round(cov, 3),
                    "kind": kind, "joined_rows": joined,
                    "left_rows": len(left_vals), "right_rows": len(right_vals),
                    "fanout": round(joined / max(len(left_vals), 1), 1)}
            # A safe join always beats a wider but unsafe one.
            if best is None or (rank, shared) > (best["_rank"], best["matched"]):
                cand["_rank"] = rank
                best = cand
    if best:
        best.pop("_rank", None)
        best["shared_dimensions"] = sorted(
            shared_dims, key=lambda d: -d["shared"])[:4]
        best["disjoint_identifiers"] = disjoint[:6]
    for d in disjoint[:6]:
        findings.append(_finding(
            f"{na} / {nb}", f'{d["left"]} vs {d["right"]}', "Unrelated Identifiers",
            f'{d["left"]} ({d["left_n"]} values) and {d["right"]} ({d["right_n"]} values) '
            "share no values at all. They are separate coding schemes, not the same "
            "entities under different names. Do not map one onto the other.",
            "High"))

    if best:
        unmatched = best["left_total"] - best["matched"]
        if best["kind"] == "many-to-many":
            findings.append(_finding(
                f"{na} / {nb}", f"{best['left']} = {best['right']}", "Unsafe Join",
                f"The only link between these boards is {best['kind']}: "
                f"'{best['left']}' repeats across {best['left_rows']} {na} rows and "
                f"'{best['right']}' across {best['right_rows']} {nb} rows. An inner "
                f"join produces {best['joined_rows']} rows from {best['left_rows']} — "
                f"a {best['fanout']}x fan-out — so any total computed across it is "
                "inflated. Aggregate each board separately and compare, rather than "
                "joining.", "Critical", 1.0))
        else:
            findings.append(_finding(
                f"{na} / {nb}", f"{best['left']} = {best['right']}", "Join Coverage",
                f"{best['kind']} join. {best['matched']} of {best['left_total']} "
                f"{na} values match ({best['coverage']:.0%}); {unmatched} do not and "
                "are dropped by an inner join.",
                severity_for(1 - best["coverage"]), 1 - best["coverage"]))
    return findings, best


# ---------------------------------------------------------------- descriptions

def describe(p: ColumnProfile) -> str:
    """A description built from the evidence, for boards that document nothing."""
    bits: List[str] = []

    lead = f"{p.inferred_meaning} " if p.inferred_meaning else ""
    kind = {"money": "Monetary amount", "quantity": "Quantity", "month": "Month name",
            "percentage": "Percentage", "date": "Date",
            "categorical": "Category", "identifier": "Identifier",
            "number": "Numeric", "text": "Free text"}[p.semantic_type]
    bits.append(f"{lead}{kind} column ({p.storage_type}).")

    if p.storage_type == "number":
        bits.append(f"{p.real} of {p.rows} rows carry a non-zero figure; "
                    f"{p.zeros} are zero, {p.nulls} blank.")
    else:
        bits.append(f"Populated in {p.filled} of {p.rows} rows ({1 - p.null_pct:.0%}).")

    real_labels = [l for l in p.labels if l]
    if real_labels and len(real_labels) <= 15:
        bits.append(f"Allowed values: {', '.join(real_labels)}.")
    elif real_labels:
        bits.append(f"{len(real_labels)} distinct values, e.g. {', '.join(real_labels[:3])}.")
    elif p.samples and p.semantic_type in ("text", "identifier"):
        bits.append(f"Examples: {', '.join(p.samples[:3])}.")

    if p.null_pct >= 0.5:
        bits.append(f"Mostly empty ({p.null_pct:.0%} missing) — treat any figure "
                    "derived from it as covering a minority of records.")
    if p.zero_pct >= HIGH_ZERO_SHARE:
        bits.append("Zero appears to stand in for absence rather than a true zero.")
    for w in p.warnings:
        bits.append(w.capitalize() + ".")
    if p.inferred_caution:
        bits.append(p.inferred_caution.rstrip(".") + ".")
    if p.cast_expr:
        bits.append(f"Cast before use: {p.cast_expr}")
    return " ".join(bits)


def attach_descriptions(profiles: Dict[str, ColumnProfile],
                        authored: Dict[str, str],
                        overrides: Optional[Dict[str, str]] = None) -> None:
    """Precedence: human override, then monday's own, then generated."""
    overrides = overrides or {}
    for name, p in profiles.items():
        generated = describe(p)
        if name in overrides:
            p.description, p.description_source = overrides[name], "override"
        elif name in authored:
            p.authored_meaning = authored[name]
            p.description = f"{authored[name]} {generated}"
            p.description_source = "monday"
        else:
            p.description, p.description_source = generated, "generated"


# ---------------------------------------------------------------- orchestration

CACHE_DIR = ".review_cache"


@dataclass
class BoardReview:
    board_id: str
    board_name: str
    table: str
    key: str
    profiles: Dict[str, ColumnProfile]
    findings: List[Dict[str, Any]]
    rows: int
    roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Workspace:
    boards: List[BoardReview]
    cross_findings: List[Dict[str, Any]] = field(default_factory=list)
    join: Optional[Dict[str, Any]] = None
    joins: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def all_findings(self) -> List[Dict[str, Any]]:
        out = []
        for b in self.boards:
            out.extend(b.findings)
        return out + self.cross_findings


def _cache_path(key: str) -> str:
    import os
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"review_{key}.json")


def load_cached(key: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_cache_path(key)) as fh:
            return json.load(fh)
    except Exception:
        return None


def save_cached(key: str, payload: Dict[str, Any]) -> None:
    try:
        with open(_cache_path(key), "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception:
        pass  # ephemeral filesystem; the in-memory cache still holds it


def drifted(cached: Dict[str, Any], profiles: Dict[str, ColumnProfile],
            threshold: float = 0.20) -> bool:
    """Has the data moved far enough to invalidate cached judgements?"""
    snap = cached.get("fill_snapshot", {})
    for name, p in profiles.items():
        before = snap.get(name)
        if before is None:
            continue
        if abs(before - (1 - p.null_pct)) > threshold:
            return True
    return False


def review_board(board_id: str, metadata: Dict[str, Any],
                 items: List[Dict[str, Any]], table: str,
                 overrides: Optional[Dict[str, str]] = None,
                 label: Optional[str] = None,
                 interpret_with: Optional[Any] = None,
                 merge_reading: Optional[Any] = None):
    """Measure, type, check and describe one board. Returns (review, typed frame)."""
    key = structure_key(board_id, metadata)
    raw = build_raw_frame(items, metadata)
    profiles = review_columns(raw, metadata)
    typed = apply_types(raw, profiles)

    display = label or metadata.get("name") or table
    roles = assign_roles(profiles)

    # Measurement is done. What remains — what a column means, and which role it
    # fills when its name matches no known cue — is not measurable, so a model
    # reads the profile once per board structure. Cached, so it never runs at
    # query time and never changes an answer between identical questions.
    cached = load_cached(key)
    reading = None
    if cached and not drifted(cached, profiles):
        overrides = {**cached.get("overrides", {}), **(overrides or {})}
        reading = cached.get("reading")
    if reading is not None and reading.pop("__incomplete__", None):
        reading = None  # cached reading was partial; interpret again
    if reading is None and interpret_with is not None:
        try:
            reading = interpret_with(display, profiles, roles)
        except Exception:
            reading = None  # interpretation is optional; measurement stands alone
    incomplete = reading.pop("__incomplete__", None) if reading else None
    if reading and merge_reading is not None:
        try:
            merge_reading(profiles, roles, reading)
        except Exception:
            pass

    attach_descriptions(profiles, descriptions_from_monday(metadata), overrides)
    findings = check_board(typed, profiles, display)
    findings += find_twin_encodings(profiles, display)

    save_cached(key, {
        "board_id": board_id,
        "table": table,
        "fill_snapshot": {n: round(1 - p.null_pct, 4) for n, p in profiles.items()},
        "descriptions": {n: p.description for n, p in profiles.items()},
        "reading": ({**reading, "__incomplete__": incomplete} if incomplete
                    else reading),
        "overrides": overrides or {},
    })

    review = BoardReview(board_id=str(board_id),
                         board_name=display,
                         table=table, key=key, profiles=profiles,
                         findings=findings, rows=len(typed), roles=roles)
    return review, typed


def build_database(frames: Dict[str, pd.DataFrame], findings: List[Dict[str, Any]]):
    """Real tables, not views, so the data is typed and queryable."""
    import duckdb
    conn = duckdb.connect(database=":memory:")
    for table, df in frames.items():
        conn.register(f"_src_{table}", df)
        conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _src_{table}')
        conn.unregister(f"_src_{table}")
    ledger = pd.DataFrame(findings) if findings else pd.DataFrame(
        columns=["board", "column", "issue_type", "description", "severity", "proportion"])
    conn.register("_src_ledger", ledger)
    conn.execute("CREATE TABLE quality_ledger AS SELECT * FROM _src_ledger")
    conn.unregister("_src_ledger")
    return conn


def agent_context(ws: Workspace) -> str:
    """Everything the agent needs to query without guessing."""
    out: List[str] = []
    for b in ws.boards:
        out.append(f'TABLE "{b.table}"  ({b.board_name}, {b.rows} rows)')
        if b.roles:
            out.append("  WHICH COLUMN TO USE FOR WHAT:")
            for role, info in b.roles.items():
                alt = (f'  (not "{info["alternatives"][0]}")'
                       if info["alternatives"] else "")
                out.append(f'    {role}: "{info["column"]}" — {info["reason"]}{alt}')
        for name, p in b.profiles.items():
            # One line per column. The measured facts live here; the second line
            # carries only what measurement cannot say — what the column means,
            # its allowed values, and how to use it. Nothing is stated twice.
            bits = [f"{p.storage_type}/{p.semantic_type}",
                    f"filled={1 - p.null_pct:.0%}"]
            if p.storage_type == "number":
                bits.append(f"nonzero={p.real}/{p.rows}")
            if p.role:
                bits.append(f"role={p.role}")
            if p.cast_expr:
                bits.append("CAST NEEDED")
            if p.storage_type == "date" and p.date_min:
                bits.append(f"range={p.date_min}..{p.date_max}")
            out.append(f'  {quote_ident(name)}  ' + "  ".join(bits))

            extra = []
            if p.inferred_meaning:
                extra.append(p.inferred_meaning.rstrip("."))
            elif p.authored_meaning:
                extra.append(p.authored_meaning.rstrip("."))
            if p.top_values and p.semantic_type in ("categorical", "month", "text"):
                shown = ", ".join(f"{v} ({n})" for v, n in p.top_values[:8])
                more = f" +{p.distinct - len(p.top_values)} more" if p.distinct > len(p.top_values) else ""
                extra.append(f"values by frequency: {shown}{more}")
            else:
                labels = [l for l in p.labels if l]
                if labels and len(labels) <= 15:
                    extra.append("values: " + ", ".join(labels))
                elif labels:
                    extra.append(f"{len(labels)} distinct values")
            if p.inferred_caution:
                extra.append(p.inferred_caution.rstrip("."))
            for w in p.warnings:
                extra.append(w)
            if p.cast_expr:
                extra.append(f"use {p.cast_expr}")
            if extra:
                out.append("      " + ". ".join(extra) + ".")
        out.append("")

    if ws.join:
        j = ws.join
        left = f'"{j["left_table"]}"."{j["left"]}"'
        right = f'"{j["right_table"]}"."{j["right"]}"'
        if j.get("kind") == "many-to-many":
            dims = j.get("shared_dimensions") or []
            out.append(
                f'DO NOT JOIN THESE TABLES. The closest shared column is {left} = '
                f'{right}, but it repeats on both sides — an inner join turns '
                f'{j["left_rows"]} rows into {j["joined_rows"]}, inflating every total '
                f'by roughly {j["fanout"]}x. There is no row-level key linking these '
                f'tables.')
            if dims:
                pairs = "; ".join(
                    f'"{d["left_table"]}"."{d["left"]}" with "{d["right_table"]}"."{d["right"]}"'
                    for d in dims)
                out.append(
                    f'  To compare them, aggregate each table separately and line the '
                    f'results up on a shared dimension: {pairs}. Use a FULL OUTER JOIN '
                    f'between the two aggregates, never between the raw tables, and '
                    f'note that the two value sets are not identical.')
            else:
                out.append('  Aggregate each table separately and present the results '
                           'side by side.')
            for d in j.get("disjoint_identifiers", []):
                out.append(
                    f'  "{d["left_table"]}"."{d["left"]}" and "{d["right_table"]}".'
                    f'"{d["right"]}" share ZERO values ({d["left_n"]} vs {d["right_n"]} '
                    "codes) — separate anonymisation schemes. They do NOT identify the "
                    "same entities. Never transform one to match the other.")
        else:
            out.append(
                f'JOIN: {left} = {right} — {j["kind"]}, {j["matched"]}/{j["left_total"]} '
                f'match ({j["coverage"]:.0%}). {j["left_total"] - j["matched"]} rows are '
                f'dropped by an inner join; say so when you join.')
        out.append("")

    high = [f for f in ws.all_findings if f["severity"] in ("Critical", "High")]
    if high:
        out.append("DATA WARNINGS (highest severity first):")
        for f in high[:25]:
            out.append(f'  [{f["severity"]}] {f["board"]} . {f["column"]}: {f["description"]}')
    return "\n".join(out)


# ---------------------------------------------------------------- verdicts

TERMINAL_NEGATIVE = ("dead", "lost", "cancelled", "canceled", "rejected",
                     "dropped", "closed lost", "struck", "abandoned")
TERMINAL_POSITIVE = ("won", "completed", "complete", "delivered", "executed",
                     "closed won", "billed", "fully billed")
ACTIVE = ("open", "ongoing", "in progress", "not started", "on hold", "pending",
          "partial", "qualified", "negotiation", "proposal")


def classify_state(state: str) -> str:
    low = str(state).strip().lower()
    if any(k in low for k in TERMINAL_NEGATIVE):
        return "abandoned"
    if any(k in low for k in TERMINAL_POSITIVE):
        return "completed"
    if any(k in low for k in ACTIVE):
        return "active"
    return "unknown"


FORECAST_CUES = ("probability", "tentative", "expected", "forecast",
                 "likelihood", "estimated", "probable")


def verdict_for(state_kind: str, missing_rate: float, forward_looking: bool = False) -> str:
    """Is a gap on this kind of record expected, or a defect?

    A field absent on an abandoned or still-active record is usually correct —
    nothing closed, so nothing to record. The same gap on a completed record is
    a defect: the work finished and the field was never filled in.
    """
    if missing_rate < 0.25:
        return "fine"
    if state_kind in ("abandoned", "active"):
        return "expected"
    if state_kind == "completed":
        # A forecast field is superseded once the record completes: a closure
        # probability on a won deal is meaningless, so its absence is correct.
        return "expected" if forward_looking else "DEFECT"
    return "unclear"


# ---------------------------------------------------------------- roles

# (role, required cues, excluded cues, semantic types, prefer cue)
ROLE_RULES = [
    ("revenue_contracted", ["amount"], ["billed", "collected", "receivable"], ["money"], "excl"),
    ("revenue_billed",     ["billed"], ["to be billed"],                      ["money"], "excl"),
    ("revenue_collected",  ["collected"], [],                                 ["money"], "excl"),
    ("receivable",         ["receivable"], [],                                ["money"], None),
    ("deal_value",         ["deal value", "deal_value"], [],                  ["money"], None),
    ("date_created",       ["created"], [],                                   ["date"], None),
    ("date_close_actual",  ["close"], ["tentative", "probable", "expected"],   ["date"], None),
    ("date_close_expected",["tentative", "expected close", "probable end"], [],["date"], None),
    ("date_start",         ["start"], [],                                     ["date"], None),
    ("date_end",           ["end"], [],                                       ["date"], None),
    ("status",             ["status"], [],                       ["categorical"], None),
    ("stage",              ["stage"], [],                        ["categorical"], None),
    ("sector",             ["sector"], [],                       ["categorical"], None),
    ("client",             ["client", "customer"], [],           ["categorical", "identifier"], None),
    ("owner",              ["owner", "personnel", "kam"], [],    ["categorical", "identifier"], None),
]


def assign_roles(profiles: Dict[str, ColumnProfile]) -> Dict[str, Dict[str, Any]]:
    """Name the column that fills each business role, with the evidence.

    Deterministic: cue match, then the best-populated candidate wins. No model
    call, so the same board always resolves the same way.
    """
    roles: Dict[str, Dict[str, Any]] = {}
    for role, required, excluded, types, prefer in ROLE_RULES:
        candidates = []
        for name, p in profiles.items():
            low = name.lower()
            if p.semantic_type not in types:
                continue
            if not any(c in low for c in required):
                continue
            if any(c in low for c in excluded):
                continue
            score = (1 if prefer and prefer in low else 0, p.real or p.filled)
            candidates.append((score, name, p))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0], reverse=True)
        _score, name, p = candidates[0]
        others = [n for _s, n, _p in candidates[1:]]
        if p.storage_type == "number":
            reason = f"{p.real} of {p.rows} rows carry a real figure"
        else:
            reason = f"populated on {p.filled} of {p.rows} rows"
        if prefer and prefer in name.lower():
            reason += "; pre-tax figure preferred for reporting"
        roles[role] = {"column": name, "reason": reason, "alternatives": others}
        p.role = role
    return roles


def compare_all(reviews: List[BoardReview],
                frames: Dict[str, pd.DataFrame]) -> Workspace:
    """Compare every pair of boards. Works for two boards or twenty."""
    findings: List[Dict[str, Any]] = []
    joins: List[Dict[str, Any]] = []
    for i, a in enumerate(reviews):
        for bd in reviews[i + 1:]:
            f, j = compare_boards(a.profiles, frames[a.table], a.board_name,
                                  bd.profiles, frames[bd.table], bd.board_name,
                                  a.table, bd.table)
            findings.extend(f)
            if j:
                joins.append(j)
    # The safest join leads; the rest are still reported.
    order = {"one-to-one": 0, "one-to-many": 1, "many-to-many": 2}
    joins.sort(key=lambda j: order.get(j.get("kind"), 3))
    return Workspace(boards=reviews, cross_findings=findings,
                     join=joins[0] if joins else None, joins=joins)
