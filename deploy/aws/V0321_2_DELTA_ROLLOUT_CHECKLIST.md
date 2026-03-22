# v0321.2 AWS Delta Rollout Checklist

## Goal

Roll out `v0321.2` on top of the already deployed AWS `v0317` environment without rebuilding the whole stack.

This checklist only covers the delta needed for:

- the independent `v0321_2` pipeline family
- revision-first artifacts and storage
- family-level prior state bootstrap
- removal of sync `global merge`

It does **not** include retrieval cutover. Retrieval stays on the old path for now.

## Live Environment Snapshot

### App / Control-plane Host

- Host: `10.60.1.243`
- SSH: `app-v0317.pem + ubuntu`
- Running services:
  - `memory-engineering-backend.service`
  - `memory-engineering-frontend.service`
- Backend path: `/opt/memory_engineering`
- Current backend launch:
  - `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
- Current frontend launch:
  - `npm run start`

### Memory Infra Host

- Host: `10.60.1.198`
- SSH: `memory-heavy.pem + ubuntu`
- Running containers:
  - `memory-redis`
  - `memory-engineering-mysql`
  - `memory-milvus`
  - `memory-neo4j`
- Current exposed ports:
  - Redis `6379`
  - MySQL `3306`
  - Milvus `19530`
  - Neo4j `7687`

### Current v0317 Runtime Facts

From `/opt/memory_engineering/.env` on `10.60.1.243`:

- `APP_VERSION=v0317`
- `DEFAULT_TASK_VERSION=v0317-Heavy`
- `APP_ROLE=control-plane`
- `WORKER_ORCHESTRATION_ENABLED=false`
- `MEMORY_EXTERNAL_SINKS_ENABLED=true`
- `MEMORY_REDIS_URL=redis://...@10.60.1.198:6379/0`
- `MEMORY_NEO4J_URI=neo4j://10.60.1.198:7687`
- `MEMORY_MILVUS_URI=http://10.60.1.198:19530`

## What Can Be Reused As-Is

These parts of the live AWS setup can stay in place for `v0321.2`:

- `10.60.1.243` backend and frontend service units
- `10.60.1.198` Redis / MySQL / Neo4j / Milvus infrastructure
- current control-plane topology
- current public frontend/backend entrypoints
- current external memory sinks network topology

## What Must Change For v0321.2

### Code Delta

Deploy code that includes:

- `services/v0321_2/`
- `services/pipeline_service.py` dispatch for `v0321.2`
- `services/face_recognition.py` support for `v0321.2`
- `memory_module/adapters.py` revision-first node publishing
- `tests/test_pipeline_memory.py` and related test coverage
- `scripts/run_v0321_2_scale_replay.py`
- `scripts/aws_staging_preflight.py`

### Config Delta

On `10.60.1.243`, update `/opt/memory_engineering/.env` so the app can run `v0321.2`.

Required delta:

- add `v0321.2` support in deployed code
- set or expose `DEFAULT_TASK_VERSION=v0321.2` only when ready to cut over
- keep `APP_ROLE=control-plane`
- keep `MEMORY_EXTERNAL_SINKS_ENABLED=true`
- keep Redis / Neo4j / Milvus URIs pointing at `10.60.1.198`

### Storage / Behavior Delta

`v0321.2` introduces:

- independent family `v0321_2`
- task-local SQLite staging under `v0321_2/state.db`
- revision-first outputs instead of sync `memory_contract`
- family-level prior state bootstrap from `TaskRecord.result["memory"]`
- `reference_media_profile_signal` for saved web images / screenshots / AI images

No retrieval path changes are part of this rollout.

## Rollout Strategy

### Phase 0: Hygiene

- clean `._*` and `.DS_Store` files from app host and infra host repos
- ensure `.gitignore` blocks these files locally
- confirm working tree only contains intentional code changes

### Phase 1: Code Sync To App Host

- sync the local `v0321.2` codebase to `10.60.1.243:/opt/memory_engineering`
- do not mix old-family artifacts with `v0321_2`
- preserve current `.env`
- keep frontend untouched unless backend API changes require frontend adjustments

### Phase 2: Pre-Cutover Validation On Host

Run on `10.60.1.243`:

- Python unit tests for:
  - `tests.test_pipeline_memory`
  - `tests.test_memory_module`
  - `tests.test_memory_adapters`
  - `tests.test_task_api`
- compile checks for:
  - `services/v0321_2`
  - `services/pipeline_service.py`
  - `memory_module/adapters.py`
- `scripts/run_v0321_2_scale_replay.py`

Pass criteria:

- no sync `global merge` path for `v0321.2`
- independent family writes under `v0321_2`
- scale replay does not show `over_segmentation_anomaly`
- no regression in API tests

### Phase 3: Controlled Runtime Enablement

Choose one of these:

1. safer path:
   - keep `DEFAULT_TASK_VERSION=v0317-Heavy`
   - trigger `v0321.2` only explicitly for validation tasks
2. cutover path:
   - switch `DEFAULT_TASK_VERSION=v0321.2`
   - restart backend service

Recommended order:

1. explicit validation task
2. inspect outputs
3. only then decide on default cutover

### Phase 4: Live Validation Against Existing Infra

Validate that `v0321.2` correctly publishes to:

- Redis on `10.60.1.198`
- Neo4j on `10.60.1.198`

For this phase, verify:

- `EventRoot/EventRevision` publish correctly
- `RelationshipRoot/RelationshipRevision` publish correctly
- `CURRENT` edges are unique and correct
- `ProfileRevision` / sidecar documents include `original_photo_ids`
- `embedded_media` does not appear as `live_presence`
- reference media updates profile only

Do **not** change retrieval or Milvus query behavior in this phase.

## Exact Validation Checklist

### Event Layer

- same-scene photos become one event or a bounded event chain
- `bj1+bj2` style cases stay in the same event
- ambiguous boundaries alone trigger boundary LLM
- stable event fields freeze after finalize

### Relationship Layer

- only affected people get recomputed
- `embedded_media` affects salience only
- `embedded_media` does not boost:
  - direct co-presence
  - one-on-one
  - contact
  - same-place interaction
- `CURRENT/HAS_REVISION/SUPERSEDES` graph stays consistent

### Profile Layer

- saved web images / screenshots / AI images become `reference_media_profile_signal`
- they update only main-user profile buckets
- they do not create ordinary events
- all profile evidence keeps `original_photo_ids`

### Bootstrap Layer

- a second task for the same user reads prior family state from `TaskRecord.result["memory"]`
- person appearances survive task boundaries
- stable original photo IDs prevent duplicate appearance inflation across re-uploads

## Cutover Risks To Watch

- app host still defaults to `v0317-Heavy`
- local dirty tree accidentally copied to server without review
- AppleDouble `._*` files reintroduced during sync
- backend service restarted before pre-cutover validation passes
- family outputs mixed with old family artifacts
- Redis / Neo4j publish succeeds but bootstrap reads the wrong family

## Recommended Minimal Delta Rollout

1. Clean repo junk on both hosts
2. Sync `v0321.2` code to `10.60.1.243`
3. Run unit tests and scale replay on `10.60.1.243`
4. Trigger one explicit `v0321.2` validation task
5. Inspect Redis / Neo4j outputs on `10.60.1.198`
6. If good, switch `DEFAULT_TASK_VERSION` to `v0321.2`
7. Restart backend service on `10.60.1.243`

## Explicit Non-Goals

Do not do these in the same rollout:

- retrieval cutover
- Milvus collection schema migration for new recall strategy
- worker orchestration redesign
- cross-family data migration
- compatibility shims for old `memory_contract`
