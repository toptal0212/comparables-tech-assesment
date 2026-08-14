"""HTTP surface: contracts, error handling, auth, limits.

These run against the real application with its real lifespan, so they also
cover startup, readiness and the index-swap-after-write path.
"""

from __future__ import annotations

import pytest

# --- health -----------------------------------------------------------------


def test_health_reports_a_loaded_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["ready"] is True
    assert body["index"]["documents"] > 0


def test_liveness_and_readiness_are_separate(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").status_code == 200


def test_metrics_exposes_latency_and_counters(client):
    client.post("/search", json={"query": "fintech in Finland"})
    body = client.get("/metrics").json()
    assert body["counters"]
    assert any("search" in k for k in body["latency_ms"])
    assert body["gauges"]["index_documents"] > 0


def test_correlation_id_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"
    assert "x-response-time-ms" in response.headers


# --- search -----------------------------------------------------------------


def test_search_returns_results_and_the_parse(client):
    body = client.post(
        "/search", json={"query": "fintech companies in Finland", "limit": 5}
    ).json()
    assert body["results"]
    assert body["parsed"]["locations"] == ["Finland"]
    assert body["total"] >= len(body["results"])
    assert body["took_ms"] > 0
    for company in body["results"]:
        assert company["location"] == "Finland"


def test_get_and_post_search_agree(client):
    query = "biotech companies in Germany"
    post = client.post("/search", json={"query": query, "limit": 5}).json()
    get = client.get("/search", params={"q": query, "limit": 5}).json()
    assert [c["id"] for c in post["results"]] == [c["id"] for c in get["results"]]


def test_explain_returns_breakdown_and_timings(client):
    body = client.post(
        "/search", json={"query": "fraud detection in Finland", "limit": 3, "explain": True}
    ).json()
    top = body["results"][0]
    assert top["score_breakdown"] is not None
    assert top["matched_on"]
    assert "retrieve" in body["timings_ms"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},                       # empty
        {"query": "x", "limit": 0},          # below range
        {"query": "x", "limit": 101},        # above range
        {"query": "x", "offset": -1},
        {"query": "x", "mode": "telepathy"},
        {"query": "x", "unexpected": True},  # extra="forbid"
        {},                                  # missing required
    ],
)
def test_invalid_search_requests_are_422(client, payload):
    response = client.post("/search", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_errors_use_one_envelope(client):
    body = client.get("/companies/999999").json()
    assert set(body) >= {"error", "request_id"}
    assert body["error"]["code"] == "not_found"


def test_relaxation_is_reported_in_the_response(client):
    body = client.post(
        "/search",
        json={"query": "UK telecom companies with fewer than 20 employees", "limit": 5},
    ).json()
    assert body["relaxation"]["applied"] is True
    assert body["relaxation"]["dropped"]
    assert body["relaxation"]["message"]


# --- companies --------------------------------------------------------------


def test_get_company(client):
    body = client.get("/companies/1").json()
    assert body["id"] == 1
    assert body["topics"]


def test_similar_excludes_the_seed(client):
    body = client.get("/companies/1/similar", params={"limit": 5}).json()
    assert body["seed"]["id"] == 1
    assert all(c["id"] != 1 for c in body["results"])


def test_similar_unknown_is_404(client):
    assert client.get("/companies/999999/similar").status_code == 404


# --- ingestion --------------------------------------------------------------


NEW_COMPANY = {
    "name": "Aurora Fraud Labs",
    "description": "Real-time platform for fraud detection.",
    "industry": "Fintech",
    "location": "Finland",
    "founded_year": 2023,
    "employee_count": 12,
    "revenue_range": "1M-10M",
}


def test_ingested_company_is_immediately_searchable(client):
    """The evolving-data requirement, end to end."""
    before = client.get("/health").json()["index"]["documents"]

    created = client.post("/companies", json=NEW_COMPANY)
    assert created.status_code == 201
    body = created.json()
    new_id = body["ids"][0]
    assert body["corpus_size"] == before + 1
    # Honest about the one thing that lags.
    assert body["semantic_indexed"] is False

    found = client.post(
        "/search",
        json={"query": "Aurora Fraud Labs", "limit": 5},
    ).json()
    assert found["results"][0]["id"] == new_id

    filtered = client.post(
        "/search",
        json={
            "query": "fintech companies in Finland with fewer than 20 employees "
                     "working on fraud detection",
            "limit": 5,
        },
    ).json()
    assert new_id in [c["id"] for c in filtered["results"]]

    assert client.delete(f"/companies/{new_id}").status_code == 204
    assert client.get("/health").json()["index"]["documents"] == before


def test_batch_ingestion(client):
    before = client.get("/health").json()["index"]["documents"]
    batch = [dict(NEW_COMPANY, name=f"Batch Co {i}") for i in range(5)]
    body = client.post("/companies", json=batch).json()
    assert body["written"] == 5
    assert client.get("/health").json()["index"]["documents"] == before + 5
    for company_id in body["ids"]:
        client.delete(f"/companies/{company_id}")


def test_deferred_refresh_then_reindex(client):
    before = client.get("/health").json()["index"]["documents"]
    body = client.post(
        "/companies", json=dict(NEW_COMPANY, name="Deferred Co"), params={"refresh": "false"}
    ).json()
    assert body["index_refreshed"] is False
    # Written to SQLite, but not yet visible to the in-memory index.
    assert client.get("/health").json()["index"]["documents"] == before

    client.post("/admin/reindex")
    assert client.get("/health").json()["index"]["documents"] == before + 1
    client.delete(f"/companies/{body['ids'][0]}")


def test_update_via_upsert(client):
    original = client.get("/companies/3").json()
    client.post("/companies", json=dict(original, name="Renamed Corp"))
    assert client.get("/companies/3").json()["name"] == "Renamed Corp"
    client.post("/companies", json=original)  # restore


def test_validate_is_a_dry_run(client):
    before = client.get("/health").json()["index"]["documents"]
    body = client.post(
        "/companies/validate",
        json={
            "company_name": "Dry Run Oy",
            "summary": "Engine for drug discovery and gene editing.",
            "country": "germany",
            "revenue": 250_000_000,
        },
    ).json()
    assert set(body["derived_topics"]) == {"drug discovery", "gene editing"}
    assert body["parsed"]["revenue_range"] == "100M-500M"
    assert body["parsed"]["location"] == "Germany"
    assert body["parsed"]["industry"] == "Biotech"  # inferred from topic
    assert client.get("/health").json()["index"]["documents"] == before


def test_delete_unknown_is_404(client):
    assert client.delete("/companies/999999").status_code == 404


def test_oversized_batch_is_rejected(client):
    body = [dict(NEW_COMPANY, name=f"C{i}") for i in range(5001)]
    response = client.post("/companies", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


# --- auth -------------------------------------------------------------------


def test_writes_require_a_key_when_configured(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_keys", "test-key-alpha,test-key-beta")
    assert settings.auth_enabled

    assert client.post("/companies", json=NEW_COMPANY).status_code == 401
    assert client.post(
        "/companies", json=NEW_COMPANY, headers={"X-API-Key": "wrong"}
    ).status_code == 401

    created = client.post(
        "/companies", json=NEW_COMPANY, headers={"X-API-Key": "test-key-alpha"}
    )
    assert created.status_code == 201
    client.delete(
        f"/companies/{created.json()['ids'][0]}", headers={"X-API-Key": "test-key-alpha"}
    )


def test_bearer_form_is_accepted(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_keys", "test-key-alpha")
    created = client.post(
        "/companies", json=NEW_COMPANY, headers={"Authorization": "Bearer test-key-alpha"}
    )
    assert created.status_code == 201
    client.delete(
        f"/companies/{created.json()['ids'][0]}", headers={"X-API-Key": "test-key-alpha"}
    )


def test_reads_stay_public_by_default(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_keys", "test-key-alpha")
    assert client.post("/search", json={"query": "fintech"}).status_code == 200


def test_reads_can_be_locked_down(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_keys", "test-key-alpha")
    monkeypatch.setattr(settings, "require_auth_for_reads", True)
    assert client.post("/search", json={"query": "fintech"}).status_code == 401
    assert client.post(
        "/search", json={"query": "fintech"}, headers={"X-API-Key": "test-key-alpha"}
    ).status_code == 200
    # Probes must never be locked out, or the platform restarts the container.
    assert client.get("/health/live").status_code == 200


# --- rate limiting ----------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rpm", 60)
    monkeypatch.setattr(settings, "rate_limit_burst", 3)

    statuses = [
        client.post("/search", json={"query": "fintech"}).status_code for _ in range(12)
    ]
    assert 429 in statuses
    limited = next(
        r for r in (client.post("/search", json={"query": "fintech"}),) if r.status_code == 429
    )
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.json()["error"]["code"] == "rate_limited"


def test_health_is_exempt_from_rate_limiting(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rpm", 60)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)

    for _ in range(20):
        assert client.get("/health/live").status_code == 200


# --- docs -------------------------------------------------------------------


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/search" in schema["paths"]
    assert "/companies/{company_id}/similar" in schema["paths"]


# --- UI ---------------------------------------------------------------------


def test_ui_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Company Search" in response.text


def test_ui_has_no_external_dependencies(client):
    """A strict-CSP or offline deployment must not need a CDN.

    Also keeps the page honest about being self-contained: an external script
    would silently become a runtime dependency of the demo.
    """
    body = client.get("/").text
    for marker in ("src=\"http", "href=\"http://", "cdn.", "googleapis", "unpkg"):
        assert marker not in body, f"external reference found: {marker}"


def test_favicon_is_answered_not_404(client):
    """Browsers request it unprompted; a 404 per page load is log noise."""
    assert client.get("/favicon.ico").status_code == 204


def test_ui_is_excluded_from_the_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
