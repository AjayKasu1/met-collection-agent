# Resumable collection ingestion

Run from the repository root after `make setup`. The default is local multilingual E5 text inference with 1,024 dimensions. Set these values together in your local `.env`:

```dotenv
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIMENSIONS=1024
```

Prepare the public-domain selection, review golden identities, and join published image vectors:

```sh
make ingest ARGS="--limit 20000 --prepare-only"
make verify-golden
make prepare-images
```

`data/objects.parquet` and `data/selection.json` reserve all eligible golden IDs. The image preparation command downloads the pinned Met SigLIP 2 Parquet shards, then writes the joined vectors and their provenance manifest. The source download is about 1.12 GB, and the local text model is about 2.24 GB. Downloads are cached under ignored `data/`; subsequent runs reuse them. No image model is run.

Keep these artifacts unchanged while a checkpoint is active. `col-008` remains in the evaluation file: Pollock's non-public-domain object is answered through `get_object`, not `search_collection`. The other eight unique golden IDs are in the public-domain selection.

## Start local inference detached

Choose an absent or empty collection. A fresh checkpoint refuses to adopt a nonempty collection. The local model runs on CPU with four threads by default. Batches of 32 are grouped by text length to reduce padding. Performance depends on the host and record lengths; the progress log records measured batch and elapsed times. Local inference has no RPM, TPM, RPD, or daily pause logic.

```sh
nohup api/.venv/bin/python -u api/scripts/ingest_collection.py \
  --limit 20000 --reuse-prepared --with-images --collection met_objects \
  --batch-size 32 --checkpoint data/local_ingest.checkpoint.json \
  >> data/local_ingest.log 2>&1 < /dev/null &
echo $! > data/local_ingest.pid
```

On macOS, this optional command keeps the machine from idle sleeping while that worker runs:

```sh
caffeinate -i -w "$(cat data/local_ingest.pid)" &
```

```sh
tail -f data/local_ingest.log
```

`local_ingestion_progress` records acknowledged counts, active elapsed seconds accumulated across resumes, and batch duration. `local_ingestion_completed` appears only after every expected ID, payload, published image vector, and declared image absence has been checked in Qdrant. `collection_indexed.wall_seconds` measures the current process's indexing stage, including model initialization and verification. Time spent stopped between attempts and earlier source downloads is separate.

`nohup` survives terminal closure, but not reboot or sleep. To stop, send SIGINT to the PID in `data/local_ingest.pid`. After the process exits, run the same launch command to resume. Do not run two workers for one checkpoint. After completion:

```sh
make verify-golden ARGS="--collection met_objects"
make publish-index ARGS="--upload"
```

Review [snapshot publication and restoration](index-artifacts.md) before choosing a destination. No writer should be running during export.

## Checkpoint guarantees

A schema-version-3 checkpoint binds the complete input digest, destination hash, collection, text provider/model/dimension, selected image artifact hash, and object count. Local checkpoints record completed Object IDs and active elapsed time; their quota field is null. An exclusive OS lock prevents concurrent use, and checkpoint writes are flushed, synchronized, and atomically replaced. Checkpoints contain no API keys.

The completed-ID set advances only after Qdrant acknowledges dense, sparse, and available image vectors together. Resume validates the run identity and every checkpointed payload before skipping those objects. Remaining records can be regrouped without losing the acknowledged set. Missing or changed records and incompatible metadata stop the run. If a process dies between an acknowledged write and its checkpoint update, that single batch may be recomputed; stable IDs make its upsert idempotent.

The stopped Gemini run's 798 vectors and payloads are preserved in `met_objects_gemini_798`, with the original checkpoint and a separate local snapshot under `data/gemini_798/`. They are not used by the local index. The original 3,072-dimensional pilot remains in `met_objects_pilot_200`. Legacy checkpoints and vector spaces must stay separate from schema-version-3 local runs.

## Optional Gemini inference

Set `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL=gemini-embedding-001`, and `EMBEDDING_DIMENSIONS=768` together. Use a new collection and checkpoint. The reduced dimension is requested through Gemini's native output dimensionality parameter and normalized before indexing. Gateway settings apply to this path when enabled.

### Gemini quotas and batching

The project's signed-in [Google AI Studio rate-limit dashboard](https://aistudio.google.com/rate-limit) was checked on September 4, 2026. It showed these free-tier limits for `gemini-embedding-001`:

| Limit | Configured budget |
| --- | ---: |
| Requests per minute | 100 |
| Input tokens per minute | 30,000 |
| Requests per day | 1,000 |
| Inputs per synchronous batch request | 100 |

RPM and RPD count embedding inputs, including individual inputs within a batch. Batching reduces HTTP overhead; it does not turn 100 embeddings into one unit of request quota. The pilot's 200 objects plus its smoke call appeared as 201 requests in the dashboard. The batch cap is also enforced in the ingestion command and corresponds to the provider constraint documented in this [SDK integration report](https://github.com/vercel/ai/issues/16101).

These are this project's observed quotas, not guaranteed limits for every account. Google's [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits) directs users to their active project dashboard. Limits are shared across the project and daily quotas reset at midnight Pacific, including daylight-saving changes. Check the dashboard before changing the command-line budgets.

Checkpoint mode sends candidate text to Gemini's native `countTokens` endpoint through the configured provider route. It fits the largest batch up to 100 inputs within the TPM allowance, reserves the measured tokens plus 10% headroom, and enforces rolling-minute RPM and TPM together. Documents above the model's 2,048-token input limit are rejected before embedding. The request counter reserves one unit per input before every attempt, including retries. Router retries are disabled in this mode so no attempt can bypass accounting.

Once RPD is exhausted, the process logs its wake time and sleeps until the next Pacific day. Provider quota errors also honor the reported retry delay; a daily quota rejection waits for the reset. Minute quota retries are bounded by `LLM_MAX_RETRIES`. Reservations are conservative: an unsuccessful request can consume local budget even when the provider did not charge it. Other clients using the same project can delay completion.

At 1,000 inputs per day, 20,000 objects need at least 20 daily quota allocations. Expect about 20 calendar days when part of today's allowance is already used. The log's `estimated_finish_at` includes the remaining daily budget and observed average tokens per object. It assumes the host stays awake, the process remains running, and no other client consumes quota. HTTP latency and failures can extend it. A fixed 30-second delay cannot remove the daily limit.

### Start a separate quota-paced run

This example assumes no embedding requests have been used today. Replace `--initial-daily-requests 0` with the dashboard's existing usage when creating a checkpoint. Persisted quota usage takes precedence on resume.

```sh
nohup api/.venv/bin/python -u api/scripts/ingest_collection.py \
  --limit 20000 --reuse-prepared --collection met_objects_gemini_768 \
  --batch-size 100 --checkpoint data/gemini_ingest.checkpoint.json \
  --requests-per-minute 100 --tokens-per-minute 30000 --requests-per-day 1000 \
  --initial-daily-requests 0 >> data/gemini_ingest.log 2>&1 < /dev/null &
echo $! > data/gemini_ingest.pid
```

Gemini checkpoint mode persists reservations before each attempt, resumes from the acknowledged input offset, and waits across daily resets. It currently supports text and sparse vectors; attaching published images in this mode is rejected explicitly. Ordinary Gemini ingestion can use prepared image vectors, but recomputes text embeddings on rerun. Do not pass `--batch-delay-seconds` with a checkpoint. Authentication, schema, and input errors stop the process with acknowledged progress retained.
