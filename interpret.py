"""Model-driven interpretation of a board's columns.

Everything measurable — types, fill rates, cardinality, quality findings — is
computed deterministically in board_review. This module handles only what
measurement cannot reach: what a column *means*, which business role it fills
when its name doesn't match any known cue, and what its values represent.

It runs once per board structure and is cached, so it never affects query
latency and never changes an answer between two identical questions.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter

SEMANTIC_TYPES = ["money", "quantity", "percentage", "date", "categorical",
                  "identifier", "month", "text", "number"]

# A role can only be filled by a column whose measured type could plausibly
# hold it. The model may suggest; the measurement decides whether it is possible.
ROLE_REQUIRES = {
    "revenue_contracted": {"money", "number"},
    "revenue_billed": {"money", "number"},
    "revenue_collected": {"money", "number"},
    "receivable": {"money", "number"},
    "deal_value": {"money", "number"},
    "quantity": {"quantity", "number"},
    "date_created": {"date"}, "date_close_actual": {"date"},
    "date_close_expected": {"date"}, "date_start": {"date"}, "date_end": {"date"},
    "status": {"categorical"}, "stage": {"categorical"},
    "sector": {"categorical"},
    "client": {"categorical", "identifier"},
    "owner": {"categorical", "identifier"},
}


class ColumnReading(BaseModel):
    """Deliberately two fields. Output length drives generation time, and the
    semantic-type correction and caution fired on a handful of columns while
    costing tokens on every one."""
    column: str
    meaning: str = Field(description="Half a sentence: what this column holds.")
    role: Optional[str] = Field(default=None, description="Business role, or null.")
    semantic_type: Optional[str] = None
    caution: Optional[str] = None


class BoardReading(BaseModel):
    columns: List[ColumnReading]


LAST_ERROR: List[str] = []


def _call(kwargs: Dict[str, Any], attempts: int = 3) -> Optional[str]:
    """One request, holding a concurrency slot, retrying on throttling.

    A swallowed 429 previously looked identical to a malformed answer, which is
    how a dozen columns silently lost their meaning.
    """
    import time
    for attempt in range(attempts):
        try:
            with _SLOTS:            # slot released before any backoff, so a
                r = _client().chat.completions.create(**kwargs)   # throttled
            return r.choices[0].message.content or ""             # call does
        except Exception as exc:                                  # not block
            msg = str(exc)                                        # the others
            if ("429" in msg or "rate" in msg.lower()) and attempt < attempts - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise
    return None


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90.0)


def _facts(profiles: Dict[str, Any], roles: Dict[str, Any]) -> str:
    resolved = {info["column"]: role for role, info in roles.items()}
    lines = []
    for name, p in profiles.items():
        bits = [f'"{name}"',
                f"declared={p.monday_type}",
                f"measured={p.storage_type}",
                f"filled={1 - p.null_pct:.0%}",
                f"distinct={p.distinct}"]
        if p.labels:
            shown = [l for l in p.labels if l][:12]
            bits.append("values=" + ", ".join(shown) + ("…" if len(p.labels) > 12 else ""))
        elif p.samples:
            bits.append("samples=" + ", ".join(p.samples[:3]))
        if name in resolved:
            bits.append(f"already_assigned_role={resolved[name]}")
        lines.append("  " + " | ".join(bits))
    return "\n".join(lines)


PROMPT = """Name each column of a business board so a SQL agent can query it.

Facts below are MEASURED from the data — never contradict them.

For each column return:
  meaning — half a sentence, what it holds. No fill rates, no restating the facts.
  role    — one of: revenue_contracted, revenue_billed, revenue_collected,
            receivable, deal_value, date_created, date_close_actual,
            date_close_expected, date_start, date_end, status, stage, sector,
            client, owner, quantity. Use null when none fits or one is already
            assigned.

Be terse. No preamble.

BOARD: {board}
{facts}

JSON: {{"columns":[{{"column":"exact name","meaning":"...","role":null}}]}}
Every column exactly once."""


CHUNK = 12

# Both boards interpret at once and each splits into chunks, so without a shared
# ceiling this fires a dozen simultaneous requests and the provider throttles.
# One global semaphore across every board keeps it inside the rate limit.
import threading
_SLOTS = threading.Semaphore(int(os.environ.get("INTERPRET_CONCURRENCY", "4")))


def read_board(board_name: str, profiles: Dict[str, Any], roles: Dict[str, Any],
               model_chain: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Interpret a board's columns.

    Output length drives latency, so a wide board is split into chunks that run
    concurrently rather than one long call. A 38-column board costs about the
    same wall-clock as a 12-column one.
    """
    names = list(profiles)
    if len(names) > CHUNK:
        from concurrent.futures import ThreadPoolExecutor
        chunks = [dict((n, profiles[n]) for n in names[i:i + CHUNK])
                  for i in range(0, len(names), CHUNK)]
        def with_retry(chunk):
            # A dropped chunk silently costs a dozen columns their meaning, and
            # the gap then gets cached. One retry, then give up loudly.
            out = _read_chunk(board_name, chunk, roles, model_chain)
            if out is None:
                out = _read_chunk(board_name, chunk, roles, model_chain)
            if out is None:
                LAST_ERROR.append(f"chunk of {len(chunk)} columns failed twice")
            return out, set(chunk)

        with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as pool:
            parts = list(pool.map(with_retry, chunks))

        merged: Dict[str, Any] = {}
        missing: set = set()
        for part, wanted in parts:
            if part:
                merged.update(part)
                missing |= wanted - set(part)
            else:
                missing |= wanted
        if missing:
            # Mark it so the next load retries instead of caching the gap.
            merged["__incomplete__"] = sorted(missing)
        return merged or None
    single = _read_chunk(board_name, profiles, roles, model_chain)
    if single is None:
        return None
    gap = set(profiles) - set(single)
    if gap:
        single["__incomplete__"] = sorted(gap)
    return single


def _read_chunk(board_name: str, profiles: Dict[str, Any], roles: Dict[str, Any],
                model_chain: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    # Naming a column is a far simpler task than multi-round SQL, so this can
    # run on a cheaper, faster model than the chat agent uses.
    chain = model_chain or [m.strip() for m in os.environ.get(
        "INTERPRET_MODEL",
        os.environ.get("MODEL_CHAIN", "z-ai/glm-5.3-flash")).split(",") if m.strip()]
    try:
        kwargs: Dict[str, Any] = {
            "model": chain[0],
            "messages": [{"role": "user", "content": PROMPT.format(
                board=board_name, facts=_facts(profiles, roles))}],
            "max_tokens": 2000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            # Naming a column needs no deliberation. This model reasons by
            # default at maximum effort, which was ~85% of its output tokens
            # and most of the wait.
            "extra_body": {"reasoning": {"effort": "low"}},
        }
        if len(chain) > 1:
            kwargs["extra_body"]["models"] = chain
        raw = _call(kwargs)
    except Exception as exc:
        LAST_ERROR.append(f"{type(exc).__name__}: {exc}")
        return None
    if raw is None:
        return None

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    try:
        reading = TypeAdapter(BoardReading).validate_python(json.loads(raw))
    except Exception:
        return None

    return {c.column: c.model_dump() for c in reading.columns}


def merge(profiles: Dict[str, Any], roles: Dict[str, Any],
          reading: Dict[str, Any]) -> int:
    """Fold the interpretation in. Measured facts always win; the model may only
    fill gaps — an unassigned role, an unclear semantic type, a missing meaning."""
    applied = 0
    taken = {info["column"] for info in roles.values()}
    for name, p in profiles.items():
        r = reading.get(name)
        if not r:
            continue
        if r.get("meaning"):
            p.inferred_meaning = r["meaning"].strip()
            applied += 1
        if r.get("caution"):
            # Kept separate from p.warnings on purpose: warnings become quality
            # findings, and a model's advice is not a measured defect.
            p.inferred_caution = r["caution"].strip()
        # Only correct a semantic type the rules left vague.
        st = r.get("semantic_type")
        if st in SEMANTIC_TYPES and p.semantic_type in ("text", "number") and st != p.semantic_type:
            p.semantic_type = st
            p.semantic_source = "inferred"
        # Only assign roles the rules could not resolve.
        role = r.get("role")
        if role and role not in ROLE_REQUIRES:
            role = None  # invented role name; only the known vocabulary is accepted
        elif role and p.semantic_type not in ROLE_REQUIRES[role]:
            role = None  # measured type cannot hold this role
        if role and role not in roles and name not in taken:
            roles[role] = {"column": name,
                           "reason": "inferred from column name and values",
                           "alternatives": [], "source": "inferred"}
            taken.add(name)
            p.role = role
    return applied
