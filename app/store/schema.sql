-- Schema for the company corpus.
--
-- SQLite is the durable source of truth. The reasoning, in full, is in
-- docs/DESIGN.md; the short version is that a single-node embedded database
-- removes a network round-trip from a 200ms budget where the query embedding
-- already costs 18ms, and a managed Postgres on a free tier would spend more
-- than that on cold starts alone. Everything derived from this table (the
-- vector matrix, the columnar filter arrays) is a rebuildable artifact.
--
-- The vector index deliberately does NOT live here. SQLite has no native vector
-- type, and serialising 384 floats per row through SQL on every query would
-- cost far more than the 2.5ms an in-memory numpy matmul takes.

-- WAL lets readers proceed while a write is in flight, which is what makes the
-- ingestion endpoint safe to call against a live service.
PRAGMA journal_mode = WAL;

-- NORMAL is the right durability/latency trade under WAL: a crash can lose the
-- last transaction, but never corrupts the database. The source of truth for
-- the corpus is the JSON file, so a lost tail is recoverable by re-ingesting.
PRAGMA synchronous = NORMAL;

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    description   TEXT    NOT NULL DEFAULT '',
    industry      TEXT    NOT NULL DEFAULT '',
    location      TEXT    NOT NULL DEFAULT '',
    founded_year  INTEGER,
    employee_count INTEGER,
    revenue_range TEXT,

    -- Numeric span of `revenue_range`, denormalised at ingest so range
    -- predicates are plain column comparisons rather than a label lookup.
    revenue_min   REAL,
    revenue_max   REAL,

    -- Canonical topics as a JSON array, for exact retrieval.
    topics        TEXT    NOT NULL DEFAULT '[]',
    -- The same topics space-joined, purely so FTS5 can index them as words.
    topics_text   TEXT    NOT NULL DEFAULT '',

    updated_at    TEXT    NOT NULL
);

-- These support the SQL-side filtering that runs alongside an FTS match. The
-- in-memory columnar masks (app/search/columns.py) handle the vector path;
-- these indexes handle the keyword path, where filtering has to happen inside
-- the query that SQLite plans.
CREATE INDEX IF NOT EXISTS idx_companies_industry  ON companies (industry);
CREATE INDEX IF NOT EXISTS idx_companies_location  ON companies (location);
CREATE INDEX IF NOT EXISTS idx_companies_founded   ON companies (founded_year);
CREATE INDEX IF NOT EXISTS idx_companies_employees ON companies (employee_count);
-- Composite: location+industry is the most common filter pair in the example
-- queries, and it is far more selective than either column alone.
CREATE INDEX IF NOT EXISTS idx_companies_loc_ind   ON companies (location, industry);

-- Full-text index over the three fields that carry lexical signal.
--
-- `content=companies` makes this an external-content table: FTS5 stores only
-- the inverted index and reads column values back from `companies`, which
-- roughly halves the database size versus duplicating the text.
--
-- unicode61 with remove_diacritics=2 matches the normalisation applied to
-- queries in app/taxonomy.normalize, so "Zürich" and "Zurich" agree.
CREATE VIRTUAL TABLE IF NOT EXISTS companies_fts USING fts5 (
    name,
    description,
    topics_text,
    content = 'companies',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Keep the FTS index in step with writes. Required for the ingestion endpoint;
-- the bulk loader bypasses these by dropping them and issuing a single
-- 'rebuild', which is an order of magnitude faster for a full corpus load.
CREATE TRIGGER IF NOT EXISTS companies_ai AFTER INSERT ON companies BEGIN
    INSERT INTO companies_fts (rowid, name, description, topics_text)
    VALUES (new.id, new.name, new.description, new.topics_text);
END;

CREATE TRIGGER IF NOT EXISTS companies_ad AFTER DELETE ON companies BEGIN
    INSERT INTO companies_fts (companies_fts, rowid, name, description, topics_text)
    VALUES ('delete', old.id, old.name, old.description, old.topics_text);
END;

CREATE TRIGGER IF NOT EXISTS companies_au AFTER UPDATE ON companies BEGIN
    INSERT INTO companies_fts (companies_fts, rowid, name, description, topics_text)
    VALUES ('delete', old.id, old.name, old.description, old.topics_text);
    INSERT INTO companies_fts (rowid, name, description, topics_text)
    VALUES (new.id, new.name, new.description, new.topics_text);
END;

-- Index provenance and bookkeeping. Lets a starting container decide whether
-- the vector matrix on disk still matches the corpus, rather than trusting that
-- a volume it just mounted is coherent.
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
