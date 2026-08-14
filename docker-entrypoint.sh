#!/usr/bin/env sh
#
# Seed the data directory from the prebuilt artifacts, then start the server.
#
# The image ships a fully-built index at $PREBUILT_DIR. $DATA_DIR is where the
# service actually reads and writes, and may or may not be a mounted volume:
#
#   no volume      -> $DATA_DIR is container-local, seeded on every start.
#                     Ingested companies live for the life of the container.
#   mounted volume -> seeded once, on the first start. After that the volume
#                     wins, so companies added through the ingestion endpoint
#                     survive restarts and redeploys.
#
# Copying rather than symlinking is deliberate: the service writes to
# companies.db (WAL, ingestion), and those writes must not land back in the
# read-only image layer.

set -eu

: "${DATA_DIR:=/data}"
: "${PREBUILT_DIR:=/opt/prebuilt}"
: "${PORT:=8000}"
: "${WEB_CONCURRENCY:=1}"

seed_if_empty() {
    if [ -f "${DATA_DIR}/companies.db" ]; then
        echo "entrypoint: existing index found at ${DATA_DIR}, leaving it alone"
        return
    fi
    if [ ! -d "${PREBUILT_DIR}" ] || [ -z "$(ls -A "${PREBUILT_DIR}" 2>/dev/null)" ]; then
        echo "entrypoint: WARNING no prebuilt index at ${PREBUILT_DIR};" \
             "the service will start but report itself not ready" >&2
        return
    fi
    echo "entrypoint: seeding ${DATA_DIR} from ${PREBUILT_DIR}"
    mkdir -p "${DATA_DIR}"
    cp -r "${PREBUILT_DIR}/." "${DATA_DIR}/"
    echo "entrypoint: seeded $(ls -1 "${DATA_DIR}" | wc -l) files"
}

case "${1:-serve}" in
    seed)
        # Seed and exit. Exists so the seeding path can be exercised on its own,
        # by tests and by an init container, without starting a server.
        seed_if_empty
        ;;
    serve)
        seed_if_empty
        # One worker by default. The service holds a ~77MB vector matrix and the
        # column arrays per process, so workers multiply memory rather than
        # sharing it — on a 512MB free-tier container, two workers is a risk of
        # being OOM-killed for throughput the event loop can already deliver.
        # Scale out with replicas, not workers, and raise WEB_CONCURRENCY only
        # where there is RAM to back it.
        echo "entrypoint: starting uvicorn on :${PORT} (workers=${WEB_CONCURRENCY})"
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --workers "${WEB_CONCURRENCY}" \
            --no-access-log \
            --timeout-keep-alive 65 \
            --proxy-headers \
            --forwarded-allow-ips '*'
        ;;
    build-index)
        shift
        exec python -m scripts.build_index --data-dir "${DATA_DIR}" "$@"
        ;;
    benchmark)
        shift
        exec python -m scripts.benchmark "$@"
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        exec "$@"
        ;;
esac
