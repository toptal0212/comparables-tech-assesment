"""BM25 keyword retrieval over SQLite FTS5.

FTS5 rather than a separate search engine. Elasticsearch or OpenSearch would
give distributed sharding and richer analysis, but both want more RAM for a
single node than this whole service is budgeted, and neither has a usable free
tier. FTS5 is already linked into SQLite, implements Okapi BM25 with per-column
weights, and answers in single-digit milliseconds at this corpus size. The
scaling story is in docs/DESIGN.md — the interface here is narrow enough that
swapping the implementation touches one file.

Two things this module has to get right:

**Query escaping.** FTS5's MATCH grammar treats bare `-`, `*`, `"`, `:`, `^`,
`AND`, `OR`, `NEAR` as operators. User text goes straight into that grammar, so
every term is quoted. Without it, a query containing "e-commerce" is parsed as
a NOT and returns wrong results — or raises, which is worse.

**Score orientation.** FTS5's `bm25()` returns *negative* numbers, more negative
meaning a better match. Every consumer here works with "higher is better", so
the sign is flipped once, at the boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import aiosqlite

from app.core.logging import get_logger
from app.store.db import Database

log = get_logger(__name__)

# Column weights for bm25(). Topics are weighted highest because they are the
# distilled signal; name above description because a company called "GridPulse
# Analytics" matching "grid" is a stronger indication than the same word
# appearing in boilerplate prose.
W_NAME = 2.0
W_DESCRIPTION = 1.0
W_TOPICS = 4.0

# Anything outside this set cannot appear inside an FTS5 quoted string safely.
_TERM_SAFE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class KeywordHit:
    company_id: int
    score: float
    rank: int


def build_match_expression(terms: list[str]) -> str | None:
    """Build a safe FTS5 MATCH expression from user terms.

    Terms are OR-ed rather than AND-ed. With an 80-word corpus vocabulary, an
    AND over several terms usually matches nothing, and BM25 already ranks
    documents that hit more terms higher — so OR gives strictly better recall at
    no cost to precision at the top of the list.
    """
    cleaned: list[str] = []
    for term in terms:
        # Strip everything the FTS5 grammar could interpret, then quote. Quoting
        # alone is not enough: an embedded double quote would close the string.
        safe = _TERM_SAFE.sub(" ", term).strip()
        if len(safe) < 2:
            continue
        cleaned.append(f'"{safe}"')
    if not cleaned:
        return None
    return " OR ".join(cleaned)


class KeywordIndex:
    """Thin BM25 retriever over the `companies_fts` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def search(
        self,
        terms: list[str],
        *,
        limit: int,
        allowed_ids: list[int] | None = None,
    ) -> list[KeywordHit]:
        """Top `limit` companies for `terms`, highest score first.

        `allowed_ids` restricts the search to a pre-filtered candidate set. It is
        passed down into SQL rather than applied afterwards so that a highly
        selective filter still returns a full page of results instead of
        whatever survives from an unfiltered top-N.
        """
        match_expr = build_match_expression(terms)
        if match_expr is None:
            return []

        sql = [
            "SELECT c.id, bm25(companies_fts, ?, ?, ?) AS score",
            "FROM companies_fts",
            "JOIN companies c ON c.id = companies_fts.rowid",
            "WHERE companies_fts MATCH ?",
        ]
        params: list[object] = [W_NAME, W_DESCRIPTION, W_TOPICS, match_expr]

        if allowed_ids is not None:
            if not allowed_ids:
                return []
            # Bound the IN list. SQLite's default parameter ceiling is 32,766;
            # beyond a few thousand ids the planner is better off scanning, and
            # the caller re-applies the mask anyway.
            if len(allowed_ids) <= 4000:
                placeholders = ",".join("?" * len(allowed_ids))
                sql.append(f"AND c.id IN ({placeholders})")
                params.extend(allowed_ids)

        sql.append("ORDER BY score LIMIT ?")  # ascending: bm25 is negative
        params.append(limit)

        async with self._db.reader() as conn:
            try:
                async with conn.execute("\n".join(sql), params) as cur:
                    rows = await cur.fetchall()
            except aiosqlite.OperationalError:
                # A malformed MATCH expression should degrade to "no lexical
                # hits", not fail the request — the vector retriever can still
                # answer it.
                log.warning("fts_query_failed", extra={"match": match_expr}, exc_info=True)
                return []

        # Flip the sign so higher is better, consistently with every other
        # retriever in the system.
        return [
            KeywordHit(company_id=row[0], score=-float(row[1]), rank=i)
            for i, row in enumerate(rows)
        ]
