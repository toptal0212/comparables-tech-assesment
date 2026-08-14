"""Controlled vocabulary for the corpus: topics, industries, locations, revenue.

Why this module exists
----------------------
Profiling `companies.json` (see docs/DESIGN.md, "Corpus analysis") turned up a
structural fact that drives the whole search design: 49,999 of the 50,000
descriptions are generated from the template

    "{modifier} {noun} for {topic}."

with 48 interchangeable `{modifier} {noun}` heads and exactly **27** distinct
topics, over a total description vocabulary of 80 words.

Two consequences:

1. The head carries no signal. "AI-powered platform" appears ~1,000 times spread
   evenly across all ten industries, so a query mentioning "AI" — as one of the
   brief's own examples does — matches thousands of unrelated companies on a
   naive bag-of-words search. Topic is the only discriminating text.

2. Because topic is low-cardinality and closed, it can be lifted out of free
   text at ingest time and treated as a structured facet. That converts the most
   important part of relevance from a fuzzy scoring problem into an exact set
   intersection, which is both more accurate and far cheaper.

Every topic in this dataset maps to exactly one industry (verified: all 27 are
100% industry-pure). We rely on that to *infer* industry from a topic — "working
on drug discovery" implies Biotech without the user saying so — but never to
override an industry the user stated explicitly.

Generalising beyond this dataset
--------------------------------
A real corpus has open-ended descriptions, so the table below would be derived
rather than hand-written: cluster the descriptions, label the clusters with an
LLM, and persist the result as the same structure. The search path does not
change — it still consumes a topic vocabulary with aliases. Only the way the
vocabulary is produced changes, which is the point of keeping it isolated here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Industries
# ---------------------------------------------------------------------------

# Canonical name -> phrases a user might type. Values are matched after
# normalisation, so casing and punctuation do not matter here.
INDUSTRY_ALIASES: dict[str, set[str]] = {
    "Fintech": {
        "fintech", "fin tech", "financial technology", "financial services",
        "finance", "financial", "banking", "bank", "payments company", "insurtech",
    },
    "Healthcare": {
        "healthcare", "health care", "health", "healthtech", "health tech",
        "medical", "medtech", "med tech", "clinical", "hospital", "care",
    },
    "Biotech": {
        "biotech", "bio tech", "biotechnology", "life science", "life sciences",
        "pharma", "pharmaceutical", "pharmaceuticals", "genomics", "bioscience",
    },
    "Technology": {
        "technology", "tech", "software", "it", "saas", "cloud", "computing",
        "developer tools", "devtools", "infrastructure software", "data",
    },
    "Energy": {
        "energy", "utilities", "utility", "power", "cleantech", "clean tech",
        "renewables", "greentech", "green tech", "grid",
    },
    "Retail": {
        "retail", "ecommerce", "e commerce", "commerce", "consumer",
        "merchandising", "shopping", "stores",
    },
    "Education": {
        "education", "edtech", "ed tech", "learning", "training", "academic",
        "schools", "university", "e learning",
    },
    "Telecom": {
        "telecom", "telecoms", "telecommunications", "telco", "carrier",
        "mobile network", "network operator", "connectivity",
    },
    "Automotive": {
        "automotive", "auto", "car", "cars", "vehicle", "vehicles", "mobility",
        "autotech",
    },
    "Logistics": {
        "logistics", "supply chain", "shipping", "freight", "transport",
        "transportation", "delivery", "fulfilment", "fulfillment", "warehousing",
    },
}

INDUSTRIES: tuple[str, ...] = tuple(INDUSTRY_ALIASES)

# ---------------------------------------------------------------------------
# Locations
#
# The dataset stores a plain country string. Aliases cover demonyms ("German"),
# ISO codes ("DE") and common informal names ("Holland"), because the example
# queries use the adjective form far more often than the country name.
# ---------------------------------------------------------------------------

LOCATION_ALIASES: dict[str, set[str]] = {
    "Finland": {"finland", "finnish", "fi", "fin", "helsinki"},
    "Germany": {"germany", "german", "de", "deu", "ger", "deutschland", "berlin", "munich"},
    "France": {"france", "french", "fr", "fra", "paris"},
    "Norway": {"norway", "norwegian", "no", "nor", "oslo"},
    "Sweden": {"sweden", "swedish", "se", "swe", "stockholm"},
    "Netherlands": {
        "netherlands", "the netherlands", "dutch", "nl", "nld", "holland",
        "amsterdam",
    },
    "USA": {
        "usa", "us", "u s", "u s a", "united states", "united states of america",
        "america", "american", "states",
    },
    "UK": {
        "uk", "u k", "united kingdom", "britain", "great britain", "british",
        "england", "english", "scotland", "wales", "gb", "gbr", "london",
    },
}

LOCATIONS: tuple[str, ...] = tuple(LOCATION_ALIASES)

# Regional shorthands that expand to several countries. "Nordic fintech" is a
# natural way to ask, and expanding it beats returning nothing.
REGION_ALIASES: dict[str, set[str]] = {
    "nordic": {"Finland", "Sweden", "Norway"},
    "nordics": {"Finland", "Sweden", "Norway"},
    "scandinavia": {"Sweden", "Norway"},
    "scandinavian": {"Sweden", "Norway"},
    "dach": {"Germany"},
    "benelux": {"Netherlands"},
    "europe": {"Finland", "Germany", "France", "Norway", "Sweden", "Netherlands", "UK"},
    "european": {"Finland", "Germany", "France", "Norway", "Sweden", "Netherlands", "UK"},
    "eu": {"Finland", "Germany", "France", "Norway", "Sweden", "Netherlands"},
}


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    """A canonical topic, the industry it implies, and how users phrase it."""

    name: str
    industry: str
    aliases: frozenset[str] = field(default_factory=frozenset)


def _t(name: str, industry: str, *aliases: str) -> Topic:
    return Topic(name=name, industry=industry, aliases=frozenset(aliases))


# The 27 topics present in the corpus. Aliases are the phrasings the example
# queries and obvious near-misses use; anything not listed still resolves via
# token-overlap and (when enabled) embedding similarity in app/nlq/topics.py, so
# this table is a fast path, not a hard gate.
TOPICS: tuple[Topic, ...] = (
    # -- Fintech
    _t("fraud detection", "Fintech",
       "fraud", "fraud prevention", "anti fraud", "fraud analytics", "risk assessment",
       "financial crime", "aml", "anti money laundering"),
    _t("banking analytics", "Fintech",
       "banking", "bank analytics", "financial analytics", "digital banking",
       "banking intelligence", "financial data"),
    _t("payments", "Fintech",
       "payment", "payment processing", "payments infrastructure", "transactions",
       "checkout", "money transfer"),
    _t("lending", "Fintech",
       "loans", "loan", "credit", "credit scoring", "underwriting", "mortgages"),
    # -- Healthcare
    _t("diagnostics", "Healthcare",
       "diagnostic", "diagnosis", "medical imaging", "screening", "pathology"),
    _t("patient monitoring", "Healthcare",
       "patient care", "remote monitoring", "vital signs", "telemedicine",
       "patient tracking"),
    _t("health data", "Healthcare",
       "health records", "ehr", "emr", "medical records", "health analytics",
       "clinical data", "health data analysis"),
    # -- Biotech
    _t("drug discovery", "Biotech",
       "drug development", "drug design", "pharmaceutical research", "compound discovery",
       "therapeutics"),
    _t("gene editing", "Biotech",
       "crispr", "genome editing", "genetic engineering", "gene therapy", "genomics"),
    _t("molecular analysis", "Biotech",
       "molecular", "proteomics", "molecular modelling", "molecular modeling",
       "compound analysis", "biotech research"),
    # -- Technology
    _t("data pipelines", "Technology",
       "data pipeline", "etl", "elt", "data engineering", "data infrastructure",
       "streaming data", "data platform"),
    _t("observability", "Technology",
       "monitoring", "telemetry", "logging", "tracing", "apm", "instrumentation"),
    # NB: bare "cloud" and bare "infrastructure" are deliberately *not* aliases.
    # Both occur in the generated description head ("cloud-native software for
    # …", "AI-powered infrastructure for …") across every industry, so treating
    # either as topic evidence mislabels ~13k companies — an Automotive firm
    # described as "cloud-native software for autonomous driving" would pick up
    # a Technology topic. An alias has to be discriminative on its own.
    _t("cloud infrastructure", "Technology",
       "cloud platform", "cloud computing", "devops", "platform engineering",
       "kubernetes", "containers", "developer tools", "serverless"),
    # -- Energy
    _t("energy forecasting", "Energy",
       "energy prediction", "load forecasting", "demand response", "power forecasting",
       "renewable forecasting"),
    _t("renewable energy", "Energy",
       "renewables", "solar", "wind", "clean energy", "green energy", "sustainability"),
    _t("smart grid", "Energy",
       "smart grids", "grid", "grid optimization", "grid optimisation",
       "grid management", "distribution network"),
    # -- Retail
    _t("pricing optimization", "Retail",
       "pricing", "pricing optimisation", "price optimization", "dynamic pricing",
       "revenue management", "markdown"),
    _t("demand forecasting", "Retail",
       "demand planning", "sales forecasting", "demand prediction"),
    _t("inventory planning", "Retail",
       "inventory", "stock management", "assortment planning", "replenishment",
       "merchandising"),
    # -- Education
    _t("personalized learning", "Education",
       "personalised learning", "adaptive learning", "individualized learning",
       "tailored learning", "learning paths"),
    _t("student analytics", "Education",
       "student data", "learning analytics", "student performance", "student outcomes",
       "academic analytics"),
    # -- Telecom
    _t("5g analytics", "Telecom",
       "5g", "five g", "5 g", "mobile analytics", "radio analytics", "5g networks"),
    _t("network optimization", "Telecom",
       "network optimisation", "network performance", "ran optimization",
       "traffic optimization", "network planning"),
    # -- Automotive
    _t("autonomous driving", "Automotive",
       "self driving", "autonomous vehicles", "adas", "driverless", "sensor fusion",
       "autonomy"),
    _t("vehicle telemetry", "Automotive",
       "connected car", "connected vehicles", "fleet telematics", "telematics",
       "vehicle data"),
    # -- Logistics
    _t("route optimization", "Logistics",
       "route optimisation", "routing", "route planning", "last mile", "dispatch",
       "fleet routing"),
    _t("supply chain visibility", "Logistics",
       "supply chain", "shipment tracking", "freight visibility", "logistics visibility",
       "track and trace"),
)

TOPIC_BY_NAME: dict[str, Topic] = {t.name: t for t in TOPICS}
TOPIC_NAMES: tuple[str, ...] = tuple(t.name for t in TOPICS)

# ---------------------------------------------------------------------------
# Revenue
#
# The dataset stores revenue as one of six bucket labels, not a number. Queries
# ask numerically ("revenue over 500M", "between 10M and 100M"), so each bucket
# is given half-open numeric bounds and matching is defined as *interval
# overlap* between the query range and the bucket range.
#
# Overlap rather than containment is a recall choice: a company in the
# 100M-500M bucket is a legitimate answer to "revenue over 200M" even though we
# cannot tell from the label whether that specific company clears 200M. The
# alternative — requiring the whole bucket to satisfy the predicate — silently
# drops correct matches. The imprecision is inherent to bucketed source data and
# is surfaced to the client in the response's `filters_applied` block.
# ---------------------------------------------------------------------------

INF = float("inf")

REVENUE_BUCKETS: dict[str, tuple[float, float]] = {
    "0-1M": (0.0, 1e6),
    "1M-10M": (1e6, 1e7),
    "10M-50M": (1e7, 5e7),
    "50M-100M": (5e7, 1e8),
    "100M-500M": (1e8, 5e8),
    "500M+": (5e8, INF),
}

REVENUE_BUCKET_ORDER: tuple[str, ...] = tuple(REVENUE_BUCKETS)


def buckets_overlapping(lo: float | None, hi: float | None) -> list[str]:
    """Revenue buckets whose numeric span intersects the half-open range [lo, hi).

    `None` means unbounded on that side.
    """
    query_lo = 0.0 if lo is None else lo
    query_hi = INF if hi is None else hi
    out = []
    for label, (blo, bhi) in REVENUE_BUCKETS.items():
        # Half-open intervals overlap iff each starts before the other ends.
        if blo < query_hi and query_lo < bhi:
            out.append(label)
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s+]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Applied identically to corpus text and query text so the two are comparable.
    "+" survives because it is meaningful in the "500M+" bucket label.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def singularize(token: str) -> str:
    """Crude English de-pluralisation, enough for a closed technical vocabulary.

    Exists so "smart grids" reaches the "smart grid" topic and "payments"
    survives as itself. A full stemmer would over-reach on this vocabulary — it
    turns "analytics" into "analyt" and merges topics we need kept apart.
    """
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("sses"):
        return token[:-2]
    # "analytics", "logistics", "payments" are canonical as-is.
    if token in _KEEP_PLURAL:
        return token
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_KEEP_PLURAL = frozenset(
    {"analytics", "logistics", "payments", "diagnostics", "renewables", "sales",
     "operations", "genomics", "proteomics", "loans", "services", "utilities"}
)


def normalize_tokens(text: str) -> list[str]:
    return [singularize(tok) for tok in normalize(text).split()]


# ---------------------------------------------------------------------------
# Reverse lookup tables, built once at import.
# ---------------------------------------------------------------------------


def _build_alias_index(groups: dict[str, set[str]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in groups.items():
        index[normalize(canonical)] = canonical
        for alias in aliases:
            index[normalize(alias)] = canonical
    return index


INDUSTRY_INDEX: dict[str, str] = _build_alias_index(INDUSTRY_ALIASES)
LOCATION_INDEX: dict[str, str] = _build_alias_index(LOCATION_ALIASES)

TOPIC_INDEX: dict[str, str] = {}
for _topic in TOPICS:
    TOPIC_INDEX[normalize(_topic.name)] = _topic.name
    for _alias in _topic.aliases:
        # First writer wins: canonical names are inserted before aliases, so an
        # alias can never shadow another topic's canonical form.
        TOPIC_INDEX.setdefault(normalize(_alias), _topic.name)

# Longest alias in tokens, so the n-gram scanner in app/nlq knows how wide a
# window it needs and prefers the most specific match.
MAX_TOPIC_NGRAM: int = max(len(normalize(k).split()) for k in TOPIC_INDEX)
MAX_LOCATION_NGRAM: int = max(len(normalize(k).split()) for k in LOCATION_INDEX)
MAX_INDUSTRY_NGRAM: int = max(len(normalize(k).split()) for k in INDUSTRY_INDEX)


def industry_for_topic(topic_name: str) -> str | None:
    topic = TOPIC_BY_NAME.get(topic_name)
    return topic.industry if topic else None


# ---------------------------------------------------------------------------
# Alias hygiene
# ---------------------------------------------------------------------------

# The generated description head is exactly "{modifier} {noun}". Both lists are
# taken from the corpus, where each term occurs ~1,000 times spread evenly
# across all ten industries — so none of them discriminates.
HEAD_MODIFIERS: tuple[str, ...] = (
    "ai powered", "data driven", "machine learning", "real time", "cloud native",
    "automated", "distributed", "predictive",
)
HEAD_NOUNS: tuple[str, ...] = (
    "platform", "solution", "engine", "software", "system", "infrastructure",
)


def _head_phrases() -> frozenset[str]:
    """Every contiguous word sequence that can occur inside a description head.

    Contiguity is the whole point. "cloud" is unsafe because the head
    "cloud native platform" contains it verbatim; "cloud platform" is safe
    because that head yields only "cloud native" and "native platform", never
    "cloud platform". A checker that merely asked "are all these words noisy?"
    would reject both and force us to drop useful aliases.
    """
    phrases: set[str] = set()
    for mod in HEAD_MODIFIERS:
        for noun in HEAD_NOUNS:
            words = f"{mod} {noun}".split()
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    phrases.add(" ".join(words[i:j]))
    return frozenset(phrases)


HEAD_PHRASES: frozenset[str] = _head_phrases()


def find_noise_aliases() -> list[tuple[str, str]]:
    """Topic aliases that would fire on the boilerplate head of any description.

    Each hit is a latent mislabelling bug. This is exactly how the bare "cloud"
    and "infrastructure" aliases were caught attaching a Technology topic to
    ~13k companies across every other industry — an Automotive company described
    as "cloud-native software for autonomous driving" was picking up
    "cloud infrastructure".

    Enforced by tests/test_taxonomy.py, alongside an empirical check that the
    extractor produces no topic/industry contradictions over the real corpus.
    """
    offenders: list[tuple[str, str]] = []
    for topic in TOPICS:
        for alias in topic.aliases:
            if " ".join(normalize_tokens(alias)) in _NORMALISED_HEAD_PHRASES:
                offenders.append((topic.name, alias))
    return offenders


# Head phrases run through the same singularisation as everything else, so the
# comparison in `find_noise_aliases` is apples to apples.
_NORMALISED_HEAD_PHRASES: frozenset[str] = frozenset(
    " ".join(normalize_tokens(p)) for p in HEAD_PHRASES
)
