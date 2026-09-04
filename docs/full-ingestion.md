# Resumable collection ingestion

Run from the repository root after `make setup`. Prepare and verify the input before indexing:

```sh
make ingest ARGS="--limit 20000 --prepare-only"
make verify-golden
```

The prepared `data/objects.parquet` and `data/selection.json` reserve all eligible golden IDs. Keep these artifacts unchanged while a checkpoint is active. `col-008` remains in the evaluation file with a note that Pollock's non-public-domain object is answered through `get_object`, not `search_collection`.

## Provider quotas and batching

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

## Start detached, watch, and resume

The default `EMBEDDING_DIMENSIONS=768` requests reduced vectors from Gemini, normalizes them, and binds both the collection schema and checkpoint to that dimension. Schema-version-1 collections such as the original 3,072-dimensional pilot remain separate. Use a new or empty collection for a new checkpoint. Snapshot export and restore also check the configured dimension; see [index artifacts](index-artifacts.md).

For a project with no embedding requests used yet today, launch:

```sh
nohup api/.venv/bin/python -u api/scripts/ingest_collection.py \
  --limit 20000 --reuse-prepared --collection met_objects \
  --batch-size 100 --checkpoint data/ingest.checkpoint.json \
  --requests-per-minute 100 --tokens-per-minute 30000 --requests-per-day 1000 \
  --initial-daily-requests 0 >> data/ingest.log 2>&1 < /dev/null &
echo $! > data/ingest.pid
```

For an existing day's usage, replace `--initial-daily-requests 0` with the number already consumed, including smoke calls. This argument is used only when creating a checkpoint. On resume, persisted quota usage takes precedence. `--batch-size 100` overrides an older, smaller `EMBEDDING_BATCH_SIZE` without editing `.env`. Omit `--batch-delay-seconds` in checkpoint mode; the two pacing modes are incompatible.

```sh
tail -f data/ingest.log
cat data/ingest.pid
```

Progress records include the acknowledged object count, batch tokens, reserved tokens, daily requests, and estimated finish time in UTC. A quota-wait event reports why processing paused and when it will resume. It is normal for the log to remain quiet during a daily wait. `ingestion_completed` is emitted only after all expected IDs and payloads have been verified in Qdrant.

`nohup` lets the job survive terminal closure. It does not keep a laptop awake or survive a reboot. After a process exits or the machine restarts, run the same launch command from the repository root to resume. Keep the same prepared artifacts, target, model, dimension, and checkpoint path. Network, authentication, input, and schema errors stop the process with acknowledged progress retained; fix the cause before resuming. Do not delete a checkpoint to retry a failed run.

## Checkpoint guarantees

The checkpoint records a SHA-256 of the complete input records, a hash of the Qdrant destination, the collection, model, dimension, schema version, total objects, next offset, completed tokens, and quota reservations. It contains no API keys. The file is flushed and synchronized before atomic replacement. An exclusive OS lock prevents concurrent use of the same checkpoint.

Every request reservation is persisted before the provider call. The offset advances only after Qdrant acknowledges both dense and sparse vectors. Resume validates the run identity and every checkpointed payload before skipping completed batches. Missing or changed records and incompatible schema metadata stop the run. A new checkpoint cannot adopt a nonempty collection.

If the process dies after Qdrant acknowledges a batch but before the checkpoint advances, that one batch may be embedded again. Stable point IDs make the repeated write idempotent. Earlier checkpointed batches are skipped, and their quota history survives the interruption. Source artifacts, checkpoints, lock files, PIDs, logs, and model caches stay under ignored `data/`.
