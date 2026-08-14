"""Build the search index from a dataset file.

Run this once at image build time or on first boot:

    python -m scripts.build_index --dataset sample_dataset/companies.json

It produces everything in `DATA_DIR`:

  companies.db     SQLite source of truth, with the FTS5 index
  vectors.npy      float32 document embedding matrix
  vector_ids.npy   row -> company id mapping
  index_meta.json  provenance, so a starting container can verify the artifacts

Splitting this out of application startup is deliberate. Embedding 50k documents
takes about two minutes; doing that on boot would mean every restart, autoscale
event and redeploy pays it again, against a brief that explicitly calls out
frequent restarts. Built ahead of time, a cold start is an mmap of a 77MB file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/build_index.py` as well as `python -m scripts.build_index`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.store.ingest import bulk_ingest  # noqa: E402

log = get_logger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the company search index.")
    parser.add_argument("--dataset", type=Path, default=settings.dataset_path)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Load SQLite and FTS only. Search still works, lexical + filters only.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing artifacts first instead of upserting into them.",
    )
    args = parser.parse_args()

    configure_logging()

    if not args.dataset.exists():
        log.error("dataset_not_found", extra={"path": str(args.dataset)})
        return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.data_dir / "companies.db"

    if args.rebuild:
        for path in (db_path, args.data_dir / "companies.db-wal",
                     args.data_dir / "companies.db-shm",
                     args.data_dir / "vectors.npy",
                     args.data_dir / "vector_ids.npy",
                     args.data_dir / "index_meta.json"):
            path.unlink(missing_ok=True)
        log.info("rebuild_cleared_artifacts")

    t0 = time.perf_counter()
    log.info("ingest_start", extra={"dataset": str(args.dataset)})
    stats = bulk_ingest(db_path, args.dataset)
    log.info("ingest_done", extra=stats.summary())

    if stats.invalid:
        log.warning(
            "ingest_had_invalid_records",
            extra={"invalid": stats.invalid, "sample": stats.errors[:5]},
        )

    if args.skip_vectors:
        log.info("skipping_vector_build")
    else:
        try:
            # Imported lazily: this pulls in onnxruntime, which is slow to load
            # and pointless when only the SQLite artifacts are wanted.
            from app.search.build_vectors import build_vector_index

            vstats = build_vector_index(db_path, args.data_dir)
            log.info("vector_build_done", extra=vstats)
        except Exception:
            # A failed embedding build must not cost us the lexical index we
            # just spent 11 seconds producing. The service detects the missing
            # vectors at startup and runs in lexical-only mode, which is a
            # degraded search but a working one.
            log.exception("vector_build_failed_continuing_lexical_only")

    total = time.perf_counter() - t0
    meta_path = args.data_dir / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["build_seconds"] = round(total, 1)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("build_complete", extra={"total_seconds": round(total, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
