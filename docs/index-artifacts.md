# Portable index artifacts

A fresh checkout can reuse verified embeddings from an existing Hugging Face dataset. The published bundle includes `objects.parquet`, a collection snapshot, a manifest, and an attributed dataset README. Image-enabled indexes also include the joined image Parquet file and its provenance manifest. Visitor chunks and their snapshot can be included together. No model calls are made during seeding.

## Export and publication

Stop ingestion writers before exporting. Set `HF_DATASET_REPO` to an existing dataset and `HF_TOKEN` to a token with write access when publishing. Repository creation and visibility changes are explicit owner actions outside this script.

```sh
# Validate and export locally. This does not upload anything.
make publish-index

# Include both indexes and explicitly upload the verified bundle.
make publish-index ARGS="--include-visitor --upload"
```

For a pilot, pass the same `--data-dir`, `--collection`, and optional `--visitor-collection` used during ingestion. Export compares every indexed ID and payload with its source artifact before creating snapshots. It rejects extra points, missing records, changed payloads, and incompatible schemas or embedding models. This prevents publishing an index that accidentally contains unrelated records.

The manifest records the format version, Qdrant version, text provider and model, sparse model, vector dimensions, point counts, image dataset/revision/model and coverage, UTC creation time, file sizes, and SHA-256 hashes. Publication uses one [Hugging Face commit](https://huggingface.co/docs/huggingface_hub/guides/upload) with an explicit file allowlist and expected parent revision. It never uploads a directory recursively. Unrelated local files, credentials, and build instructions are excluded.

Manifest format version 3 requires collection schema version 3. Both the snapshot's collection metadata and the manifest record the text provider, model, and dimension, plus the image source dataset, revision, model, and dimension when present. Export checks these fields against the configured text identity and the joined image manifest, then compares every image vector to its published source. Missing image vectors are valid only for the explicitly declared missing IDs.

Restore rejects a text provider, model, or dimension mismatch before uploading a snapshot. It validates artifact hashes and declared image coverage before restore, then checks the restored metadata, payloads, and every image vector afterward. Version-1 and version-2 Gemini indexes remain separate; this workflow does not upgrade or relabel them. Query embedding uses the same model compatibility gate.

Export currently supports a collection whose shards are all on one Qdrant node. A distributed collection needs snapshots from each node and is rejected. For Qdrant's snapshot scope and recovery constraints, see its [snapshot documentation](https://qdrant.tech/documentation/operations/snapshots/).

## Restore without embedding

```sh
make seed
# A local exported bundle can also be restored:
make seed ARGS="--bundle data/bundle --collection met_objects_restored"
```

Pass `--revision` with a dataset commit hash to pin a specific release. The revision is resolved once to an immutable Hub commit. Every file is downloaded from that same commit. Public downloads explicitly disable implicit token discovery; a configured `HF_TOKEN` is passed directly for private datasets. Only files declared in the typed manifest are downloaded. The script validates all hashes, sizes, source records, point counts, and the configured embedding provider, model, and dimension before making restore requests.

The target server must use the same Qdrant minor version and an equal or newer patch version. This project deliberately supports that conservative subset of Qdrant's migration compatibility. Target collections must be absent and distinct. Seed never overwrites an existing collection. It uploads each snapshot with `priority=snapshot`, waits for recovery, and verifies the restored schema and complete payloads against the source artifacts.

A network failure during restoration can leave a newly created collection or one completed index from a two-index bundle. Inspect that target and choose unused names for a retry. The script does not delete collections automatically. Successful restoration also installs the source artifacts in `DATA_DIR` so golden verification and later exports use the same records.

Hashes protect transfer integrity; they do not make an untrusted dataset authentic. Use a dataset and pinned revision that you control or have reviewed. Publishing visitor-page extracts does not change their ownership or imply that visitor website text has the collection dataset's CC0 status.
