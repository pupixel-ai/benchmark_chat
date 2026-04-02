# Memory Engineering — Agent Architecture Overview

This document describes the upstream agent pipeline used by Memory Engineering to transform raw photo-album data into structured user profiles. The pipeline is implemented primarily under `services/v0325/` and organized into two major stages: **Core Agents (LP2)** and **Profile Agent with Downstream Audit (LP3)**.

---

## Pipeline Execution Order

```
VP1 (VLM Observations)
 → LP1 (Event Extraction)
   → Person Screening
     → Primary Person Agent
       → Relationship Dossier Construction
         → Relationship Inference Agent
           → Group Detection
             → LP3 Profile Agent (7 Domain Groups × Field-level COT + 6 Tools)
               → Downstream Audit (Critic → Judge → Three-way Backflow)
```

Entry point: `V0325PipelineFamily._run_from_lp1_payload()` in `services/v0325/pipeline.py:2041`.

---

## Part 1 — Core Agents (LP2 Layer)

Code location: `services/v0325/lp3_core/`

### 1a. Person Screening + Primary Person Agent

**Files**: `person_screening.py`, `primary_person.py`

#### Person Screening (`screen_people`)

Iterates over all persons in `face_db` and classifies them by analyzing VLM observation statistics:

| Condition | `person_kind` | `memory_value` |
|---|---|---|
| `mediated_ratio >= 0.8` (poster, screenshot, TV) | `mediated_person` | `block` |
| `service_ratio >= 0.7` (waiter, cashier, staff) | `service_person` | `block` |
| `photo_count <= 1` and `group_only_ratio >= 0.8` | `incidental_person` | `low_value` |
| `photo_count >= 8` | `real_person` | `core` |

Blocked persons are excluded from all downstream processing.

#### Primary Person Identification (`analyze_primary_person_with_reflection`)

A four-step process with self-reflection:

**Step 1 — Signal Collection** (`_collect_candidate_signals`):
For each non-blocked candidate, collects six signal dimensions from VLM results and events:
- `photo_count` — raw appearance frequency
- `event_count` — number of events participated in
- `protagonist_label_count` — VLM 【主角】 tag hits
- `selfie_count` — selfie scene detections
- `identity_anchor_count` — ID card, badge, student ID detections
- `photographed_subject_ratio` — ratio of "being photographed by others" signals

**Step 2 — Weighted Ranking** (`_candidate_sort_key`):
```
score = selfie × 4.0
      + identity_anchor × 3.0
      + protagonist_label × 2.2
      + event × 1.1
      + photo × 0.15
      − photographed_subject_ratio × 3.5
```

**Step 3 — LLM Judgement** (`_run_llm_primary_judgement`):
Sends top-5 candidate statistics table + key event clues to LLM. Expected JSON response:
```json
{
  "primary_person_id": "Person_xxx",
  "confidence": 0-100,
  "key_signals": [{"signal": "...", "points_to": "Person_xxx", "detail": "..."}],
  "conflicts": ["..."],
  "runner_up": {"person_id": "...", "relationship_guess": "...", "why_not": "..."},
  "reasoning": "..."
}
```

**Step 4 — Decision + Reflection** (`_build_primary_decision` → `_reflect_primary_decision` → `_apply_primary_reflection`):
- Trusts LLM result when the selected person exists in the candidate pool.
- Falls back to rule-based decision when LLM is unavailable or returns invalid output.
- Multiple `photographer_mode` fallback triggers:
  - Top candidate `photographed_subject_ratio >= 0.6` with zero selfies/identity anchors
  - No stable anchor signals (no selfies, no identity anchors, no protagonist labels)
  - Ambiguous top candidates (score gap <= 0.6)
  - Top candidate is `low_value` or non-`real_person`
- Reflection layer checks for misidentification risks and can override to `photographer_mode`.

### 1b. Relationship Inference Agent

**File**: `relationships.py`

#### Dossier Construction (`build_relationship_dossiers`)

For every non-blocked, non-primary person, builds a `RelationshipDossier` containing:
- `photo_count`, `time_span_days`, `recent_gap_days`, `monthly_frequency`
- `scene_profile` — scenes, `private_scene_ratio`, `dominant_scene_ratio`, `with_user_only`
- `intimate_signals` — strong contact signals (selfie_together, hug, kiss, holding_hands, etc.)
- `shared_events` — list of co-occurring events

#### Relationship Type Specs

Eight relationship types, each defined as a `RelationshipTypeSpec` with structured reasoning constraints:

| Type | Strong Evidence | Blocker Evidence | Downgrade Target |
|---|---|---|---|
| `family` | household scene, caregiving, generational structure | activity-only child, random co-frame, service role | `close_friend` |
| `romantic` | intimate contact, stable 1v1 private scenes | group-dominated, single highlight event | `close_friend` |
| `bestie` | high frequency, multi-scene, strong interaction | functional-scene dominated | `close_friend` |
| `close_friend` | mid-high repeated interaction, multi-scene | single co-frame, no interaction | `friend` |
| `friend` | repeated appearance, light-medium interaction | pure group encounters | `acquaintance` |
| `classmate_colleague` | functional-scene dominated | high private-scene ratio | `friend` |
| `activity_buddy` | single-activity repeated appearance | cross-life-scene stable co-occurrence | `friend` |
| `acquaintance` | low frequency, group dominated, weak exclusivity | 1v1 private scenes | — |

Each spec includes `cot_steps` (chain-of-thought instructions) and `reflection_questions` for the LLM to self-check.

#### Relationship Inference (`infer_relationships_from_dossiers`)

Sends dossier evidence to LLM for type determination, guided by the per-type COT steps.

### 1c. Group Detection

**File**: `groups.py`

Detects social groups by finding events where multiple relationship-eligible persons co-occur. Infers group type (sorority, lab, team, club, friend_group) from event keywords. Confidence is scored based on member count, participant overlap, photo count, and event confidence.

---

## Part 2 — Profile Agent (LP3 Layer)

Code location: `services/v0325/lp3_core/profile_agent.py`, `profile_fields.py`, `profile_tools.py`

### Domain Group Architecture

The Profile Agent processes fields through **7 domain groups**, executed sequentially:

| # | Domain Group | Example Fields |
|---|---|---|
| 1 | Foundation & Social Identity | name, gender, age_range, role, race, nationality, education, career, career_phase, professional_dedication, language_culture |
| 2 | Wealth & Consumption | asset_level, spending_style, brand_preference, income_model, signature_items, spending_shift |
| 3 | Spatio-Temporal Habits | location_anchors, mobility_pattern, cross_border, life_rhythm, event_cycles, sleep_pattern, phase_change, current_displacement, recent_habits |
| 4 | Relationships & Household | intimate_partner, close_circle_size, social_groups, pets, parenting, living_situation |
| 5 | Taste & Interests | interests, frequent_activities, solo_vs_social, fitness_level, diet_mode, life_events, recent_interests |
| 6 | Visual Expression | attitude_style, aesthetic_tendency, visual_creation_style, current_mood, social_energy |
| 7 | Semantic Expression | personality_mbti, morality, philosophy, mental_state, motivation_shift, stress_signal |

#### Role of Domain Groups

Domain groups serve three purposes in the pipeline:

**1. Semantic Isolation for LLM Reasoning**

Each domain group runs an independent `_run_domain()` cycle. Fields from different domains never appear in the same LLM batch request. The prompt explicitly scopes the LLM to the current domain:

```
# Domain: Wealth & Consumption
Current fields: [asset_level, spending_style, brand_preference]
```

This prevents cross-domain evidence interference — the LLM does not need to reason about geography patterns and personality traits in the same prompt context.

**2. Ordered Context Accumulation**

After each field is resolved, `resolved_facts_summary` is updated and carried forward. Later domain groups receive a `Resolved Facts Summary` section in their prompt containing all previously resolved fields.

The ordering is intentional: Foundation & Social Identity (name, gender, role, education, career) resolves first, so downstream domains like Wealth & Consumption and Taste & Interests can reference established identity facts when making inferences.

**3. Batch Granularity Scope**

Within each domain group, fields are split into LLM batches by risk level:
- **P0 fields** (high risk: role, race, nationality, education, career, asset_level, etc.) — max 2 per batch
- **P1 fields** (standard: name, gender, age_range, career_phase, etc.) — max 5 per batch

This splitting only operates within a single domain group. A batch will never mix fields from different domains.

### Per-Field Processing Pipeline

For each field within a domain group, four tool functions are called in sequence before LLM invocation:

```
fetch_field_evidence → check_subject_ownership → analyze_evidence_stats → find_counter_evidence
```

**1. `fetch_field_evidence`** — Extracts evidence from allowed sources (vlm, event, feature, relationship, group) for the specific field.

**2. `check_subject_ownership`** — Verifies whether evidence belongs to the album owner (not someone else's belongings, screenshot text, environment, or co-occurring persons).

**3. `analyze_evidence_stats`** — Computes field-specific statistics: brand top lists with source counts, city-level location bucketing, topic novelty scores, work signal summaries, etc.

**4. `find_counter_evidence`** — Searches for evidence contradicting the draft value.

### Field Specifications (`FieldSpec`)

Each of the ~40 fields has a `FieldSpec` defining:

| Attribute | Purpose |
|---|---|
| `risk_level` | P0 (high risk, small batch) or P1 (standard) |
| `allowed_sources` | Which evidence types this field can use (vlm, event, feature) |
| `strong_evidence` | What constitutes strong support for a non-null value |
| `hard_blocks` | Conditions that force null output |
| `cot_steps` | Field-specific chain-of-thought reasoning instructions |
| `owner_resolution_steps` | Steps to verify evidence belongs to the protagonist |
| `time_reasoning_steps` | Long-term vs short-term layer determination |
| `counter_evidence_checks` | What counter-evidence to look for |
| `null_preferred_when` | Conditions where null is the safer output |
| `reflection_questions` | Self-check questions after draft |
| `requires_social_media` | If true, silent null when no social media evidence exists |

20+ fields have custom COT overrides in `FIELD_COT_OVERRIDES` (e.g., education requires cross-event campus evidence; brand_preference requires cross-event brand repetition with ownership verification).

### Deterministic Fields

Some fields bypass LLM entirely and are computed from upstream agent outputs:
- `intimate_partner` — reads directly from relationship layer's `romantic` result
- `close_circle_size` — computed from relationship counts
- `social_groups` — reads from `GroupArtifact` outputs
- `sleep_pattern` — computed from late-night event ratio
- `living_situation` — inferred from events + relationships

### LLM Batch Prompt Structure

Each LLM call receives a structured prompt containing:

```
# Role: 结构化画像的字段裁决 agent
# Domain: {domain_name}
# Resolved Facts Summary (from earlier domains)
# Reasoning Protocol (4-step)
# Field Units (for each field in batch):
  - Risk level
  - High Weight Signals
  - COT Steps
  - Owner Resolution steps
  - Time Reasoning steps
  - Counter Evidence Checks
  - Evidence Summary (from tools)
  - Stats Summary (from tools)
  - Ownership Summary (from tools)
  - Counter Summary (from tools)
# Output Contract (JSON schema)
```

---

## Part 3 — Downstream Audit (Judge Agent + Critic Agent)

Code location: `services/v0325/lp3_core/downstream_audit.py`

The downstream audit is a two-agent review system that runs after the Profile Agent completes. It operates on three dimensions: protagonist, relationships, and profile fields.

### Adapter Layer (`profile_agent_adapter.py`)

Converts pipeline outputs (primary_decision, relationships, structured_profile) into a standardized `extractor_output` format with dimension-mapped tags that the external `profile_agent` package expects.

### Critic Agent

```python
critic = CriticAgent(storage)
critic_result = critic.run(extractor_output)
```

For each tag in the extractor output, the Critic raises challenges — questioning whether the evidence truly supports the claim, whether ownership is correct, whether the relationship type is too high, etc.

### Judge Agent

```python
judge = JudgeAgent(storage)
if critic_result.has_challenges():
    judge_result = judge.run(extractor_output, critic_output, album_id=album_id)
else:
    judge_result = judge.run_no_challenges(extractor_output)
```

The Judge reviews each Critic challenge and renders a verdict:
- **`accept`** — the original value stands
- **`nullify`** — the value is invalid, set to null
- **`downgrade`** — the value should be demoted (e.g., long_term → short_term, or relationship type lowered)

### Three-way Backflow

Judge decisions flow back to modify the pipeline outputs:

| Backflow Target | Function | `nullify` Effect | `downgrade` Effect |
|---|---|---|---|
| **Protagonist** | `apply_downstream_protagonist_backflow` | Revert to `photographer_mode`, clear `primary_person_id` | — |
| **Relationships** | `apply_downstream_relationship_backflow` | Drop the relationship entirely (retention = drop) | Lower to `downgrade_target` type, cap confidence at 0.7 |
| **Profile Fields** | `apply_downstream_profile_backflow` | Set value = null, confidence = 0.0 | Annotate as short_term downgrade |

All backflow actions append `constraint_notes` (e.g., `downstream_judge:nullify:reason`) to the evidence record for traceability.

### Audit Report

The final audit report includes per-dimension statistics:
- `total_audited_tags`
- `challenged_count` — tags the Critic questioned
- `accepted_count` — tags the Judge accepted
- `downgraded_count` — tags the Judge downgraded
- `rejected_count` — tags the Judge nullified
- `not_audited_count` — tags that were not covered by the audit (e.g., `photographer_mode` protagonist, unsupported relationship types)

---

## File Reference

| File | Role |
|---|---|
| `services/v0325/pipeline.py` | Pipeline orchestrator (V0325PipelineFamily) |
| `services/v0325/lp3_core/person_screening.py` | Person screening (block/low_value/core classification) |
| `services/v0325/lp3_core/primary_person.py` | Primary person identification with reflection |
| `services/v0325/lp3_core/relationships.py` | Relationship dossier construction and type inference |
| `services/v0325/lp3_core/groups.py` | Social group detection |
| `services/v0325/lp3_core/profile_agent.py` | Profile Agent (domain-group orchestration, LLM batch calls) |
| `services/v0325/lp3_core/profile_fields.py` | Field specifications (FieldSpec + COT overrides) |
| `services/v0325/lp3_core/profile_tools.py` | Six tool functions (evidence, ownership, stats, counter, resolved facts, metadata) |
| `services/v0325/lp3_core/downstream_audit.py` | Critic/Judge audit + three-way backflow |
| `services/v0325/lp3_core/profile_agent_adapter.py` | Adapter between pipeline and external profile_agent package |
| `services/v0325/lp3_core/types.py` | Shared dataclasses (MemoryState, FieldSpec, RelationshipDossier, etc.) |
| `services/v0325/lp3_core/evidence_utils.py` | Evidence payload construction utilities |
| `services/v0325/lp3_core/consistency_checker.py` | Cross-field consistency checking |
| `services/v0325/profile_compaction.py` | Structured profile compaction for output |
