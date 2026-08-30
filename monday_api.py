"""monday.com API access.

Every call goes through one gateway that refuses anything but a read query,
so the "monday.com — read only" constraint is enforced by the code rather
than left as a promise.
"""
import os
import re
import time
import requests
from typing import Dict, Any, List, Optional

API_URL = "https://api.monday.com/v2"
API_VERSION = "2024-01"
TIMEOUT = 90
MAX_RETRIES = 3


class MondayWriteBlocked(RuntimeError):
    """Raised when a request would modify monday.com data."""


def _headers() -> Dict[str, str]:
    token = os.environ.get("MONDAY_API_TOKEN")
    if not token:
        raise ValueError("MONDAY_API_TOKEN environment variable not set")
    return {
        "Authorization": token,
        "API-Version": API_VERSION,
        "Content-Type": "application/json",
    }


def _assert_read_only(query: str) -> None:
    """Reject any GraphQL document that is not a plain read query.

    Anonymous (`{ ... }`) and named (`query Foo { ... }`) operations pass.
    Anything else -- mutation, subscription -- is refused before it is sent.
    """
    stripped = re.sub(r"#[^\n]*", "", query).strip()
    if not stripped:
        raise MondayWriteBlocked("Empty GraphQL document")
    if not (stripped.startswith("{") or re.match(r"^query\b", stripped)):
        raise MondayWriteBlocked(
            f"Refusing non-query operation: {stripped[:40]!r}. "
            "This application is read-only against monday.com."
        )


def graphql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The single entry point for every monday.com call."""
    _assert_read_only(query)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                json={"query": query, "variables": variables or {}},
                headers=_headers(),
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = f"Network error contacting monday.com: {exc}"
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429:
            time.sleep(2 ** attempt * 2)
            last_error = "Rate limited by monday.com"
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            last_error = f"monday.com returned {response.status_code}"
            continue
        if response.status_code != 200:
            raise RuntimeError(f"monday.com returned {response.status_code}: {response.text[:300]}")

        payload = response.json()
        if "errors" in payload:
            text = str(payload["errors"])
            # monday returns complexity-budget and rate-limit failures as HTTP 200
            # with an errors key, so the status-code backoff above never sees them.
            if any(k in text for k in ("Complexity", "complexity", "Rate limit",
                                       "rate limit", "RATE_LIMIT", "budget")):
                time.sleep(2 ** attempt * 5)
                last_error = f"monday.com throttled: {text[:200]}"
                continue
            raise RuntimeError(f"monday.com GraphQL error: {payload['errors']}")
        return payload.get("data", {})

    raise RuntimeError(f"monday.com request failed after {MAX_RETRIES} attempts: {last_error}")


_BOARDS_QUERY = """
query ($limit: Int!) {
  boards(limit: $limit, order_by: created_at) {
    id
    name
    items_count
  }
}
"""

_METADATA_QUERY = """
query ($boardId: [ID!]) {
  boards(ids: $boardId) {
    id
    name
    description
    columns { id title type description settings_str }
  }
}
"""

_ITEMS_QUERY = """
query ($boardId: [ID!], $cursor: String) {
  boards(ids: $boardId) {
    items_page(limit: 500, cursor: $cursor) {
      cursor
      items {
        id
        name
        column_values { id type text value }
      }
    }
  }
}
"""


def list_boards(limit: int = 50) -> List[Dict[str, Any]]:
    """Boards visible to this token. Used when board IDs are not configured."""
    data = graphql(_BOARDS_QUERY, {"limit": limit})
    return data.get("boards") or []


def fetch_board_metadata(board_id: str) -> Dict[str, Any]:
    """Board name, description, and full column definitions."""
    data = graphql(_METADATA_QUERY, {"boardId": [str(board_id)]})
    boards = data.get("boards") or []
    if not boards:
        raise RuntimeError(f"Board {board_id} not found, or the token cannot see it")
    return boards[0]


def fetch_board_items(board_id: str) -> List[Dict[str, Any]]:
    """Every item on the board, following pagination to the end."""
    all_items: List[Dict[str, Any]] = []
    cursor = None

    while True:
        variables: Dict[str, Any] = {"boardId": [str(board_id)]}
        if cursor:
            variables["cursor"] = cursor

        data = graphql(_ITEMS_QUERY, variables)
        boards = data.get("boards") or []
        if not boards:
            break

        page = boards[0].get("items_page") or {}
        all_items.extend(page.get("items") or [])

        cursor = page.get("cursor")
        if not cursor:
            break
        time.sleep(0.1)

    return all_items
