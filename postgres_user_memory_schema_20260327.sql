-- Memory Engineering v0327-db
-- PostgreSQL DDL draft aligned with the current task-only implementation.
--
-- Key rules:
-- 1. task_id is the only result identity.
-- 2. dataset_id groups rerun tasks that share the same fixed photo set.
-- 3. No run_id / pipeline_runs / stage_runs model is used in v0327-db.
-- 4. Numeric version filters live on tasks and task_stage_records.
-- 5. This DDL mirrors backend/models.py + backend/memory_models.py.
-- 6. Current workflow stays unchanged; DB writes are additive mirrors.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Core control-plane tables with the datasets/tasks cycle broken by ALTER TABLE
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(256) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64),
    dataset_id INTEGER,
    dataset_fingerprint VARCHAR(64),
    version VARCHAR(16),
    pipeline_version INTEGER,
    pipeline_channel VARCHAR(32),
    face_version INTEGER,
    vlm_version INTEGER,
    lp1_version INTEGER,
    lp2_version INTEGER,
    lp3_version INTEGER,
    judge_version INTEGER,
    status VARCHAR(32) NOT NULL,
    stage VARCHAR(64) NOT NULL,
    upload_count INTEGER NOT NULL DEFAULT 0,
    task_dir VARCHAR(512) NOT NULL,
    progress JSON,
    uploads JSON,
    options JSON,
    result JSON,
    result_summary JSON,
    asset_manifest JSON,
    error TEXT,
    worker_instance_id VARCHAR(64),
    worker_private_ip VARCHAR(64),
    worker_status VARCHAR(32),
    delete_state VARCHAR(32),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITHOUT TIME ZONE,
    deleted_at TIMESTAMP WITHOUT TIME ZONE,
    last_worker_sync_at TIMESTAMP WITHOUT TIME ZONE,
    FOREIGN KEY(user_id) REFERENCES users (user_id)
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    dataset_fingerprint VARCHAR(64) NOT NULL,
    photo_count INTEGER NOT NULL,
    source_hashes_json JSON,
    first_task_id VARCHAR(64),
    latest_task_id VARCHAR(64),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    CONSTRAINT uq_dataset_user_fingerprint UNIQUE (user_id, dataset_fingerprint),
    FOREIGN KEY(user_id) REFERENCES users (user_id)
);

ALTER TABLE tasks
    ADD CONSTRAINT fk_tasks_dataset_id
    FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id);

ALTER TABLE datasets
    ADD CONSTRAINT fk_datasets_first_task_id
    FOREIGN KEY(first_task_id) REFERENCES tasks (task_id);

ALTER TABLE datasets
    ADD CONSTRAINT fk_datasets_latest_task_id
    FOREIGN KEY(latest_task_id) REFERENCES tasks (task_id);

-- TABLE: agent_runs

CREATE TABLE agent_runs (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64), 
	agent_kind VARCHAR(128), 
	agent_name VARCHAR(255), 
	model_name VARCHAR(255), 
	prompt_hash VARCHAR(128), 
	status VARCHAR(32), 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: artifacts

CREATE TABLE artifacts (
	artifact_id VARCHAR(128) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	relative_path VARCHAR(512) NOT NULL, 
	stage VARCHAR(64), 
	content_type VARCHAR(255), 
	size_bytes INTEGER NOT NULL, 
	sha256 VARCHAR(64), 
	storage_backend VARCHAR(32) NOT NULL, 
	object_key VARCHAR(512), 
	asset_url VARCHAR(512), 
	metadata JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (artifact_id), 
	CONSTRAINT uq_artifact_task_path UNIQUE (task_id, relative_path), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: binary_assets

CREATE TABLE binary_assets (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	asset_kind VARCHAR(64) NOT NULL, 
	relative_path VARCHAR(512) NOT NULL, 
	mime_type VARCHAR(255), 
	size_bytes INTEGER NOT NULL, 
	sha256 VARCHAR(64), 
	storage_backend VARCHAR(32), 
	object_key VARCHAR(512), 
	asset_url VARCHAR(512), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_binary_asset_task_path UNIQUE (task_id, relative_path), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: event_revisions

CREATE TABLE event_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	event_id VARCHAR(64) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	title VARCHAR(255), 
	date VARCHAR(64), 
	time_range JSON, 
	duration JSON, 
	type VARCHAR(128), 
	location JSON, 
	description TEXT, 
	photo_count INTEGER, 
	confidence FLOAT, 
	reason TEXT, 
	narrative TEXT, 
	narrative_synthesis TEXT, 
	tags_json JSON, 
	social_dynamics_json JSON, 
	persona_evidence_json JSON, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_event_revision_task_event UNIQUE (task_id, event_id, revision_no), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: event_roots

CREATE TABLE event_roots (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	event_id VARCHAR(64) NOT NULL, 
	current_revision_no INTEGER NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_event_root_task_event UNIQUE (task_id, event_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: face_recognition_image_policies

CREATE TABLE face_recognition_image_policies (
	policy_id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	source_hash VARCHAR(128) NOT NULL, 
	is_abandoned BOOLEAN NOT NULL, 
	last_task_id VARCHAR(64), 
	last_image_id VARCHAR(64), 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (policy_id), 
	CONSTRAINT uq_face_policy_user_hash UNIQUE (user_id, source_hash), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: face_reviews

CREATE TABLE face_reviews (
	review_id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	face_id VARCHAR(64) NOT NULL, 
	image_id VARCHAR(64) NOT NULL, 
	person_id VARCHAR(64), 
	source_hash VARCHAR(128), 
	is_inaccurate BOOLEAN NOT NULL, 
	comment_text TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (review_id), 
	CONSTRAINT uq_face_review_user_task_face UNIQUE (user_id, task_id, face_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: ground_truth_revisions

CREATE TABLE ground_truth_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	gt_revision_id VARCHAR(128) NOT NULL, 
	name VARCHAR(255), 
	status VARCHAR(32), 
	note TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: group_revisions

CREATE TABLE group_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	group_id VARCHAR(128) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	group_type_candidate VARCHAR(128), 
	confidence FLOAT, 
	reason TEXT, 
	strong_evidence_refs_json JSON, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_group_revision UNIQUE (task_id, group_id, revision_no), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: group_roots

CREATE TABLE group_roots (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	group_id VARCHAR(128) NOT NULL, 
	current_revision_no INTEGER NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_group_root_task_group UNIQUE (task_id, group_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: judge_decision_revisions

CREATE TABLE judge_decision_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	judge_type VARCHAR(128), 
	target_id VARCHAR(255), 
	decision_json JSON, 
	reasoning_text TEXT, 
	score FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: object_links

CREATE TABLE object_links (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	from_id VARCHAR(64) NOT NULL, 
	relation_type VARCHAR(64) NOT NULL, 
	to_id VARCHAR(64) NOT NULL, 
	weight FLOAT, 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: object_registry

CREATE TABLE object_registry (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64), 
	dataset_id INTEGER, 
	object_type VARCHAR(64) NOT NULL, 
	semantic_id VARCHAR(255), 
	table_name VARCHAR(128), 
	parent_id VARCHAR(64), 
	root_id VARCHAR(64), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_object_registry_task_semantic UNIQUE (task_id, object_type, semantic_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: person_revisions

CREATE TABLE person_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	person_id VARCHAR(64) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	snapshot_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_person_revision UNIQUE (task_id, person_id, revision_no), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: persons

CREATE TABLE persons (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	person_id VARCHAR(64) NOT NULL, 
	canonical_name VARCHAR(255), 
	is_primary_person BOOLEAN NOT NULL, 
	photo_count INTEGER NOT NULL, 
	face_count INTEGER NOT NULL, 
	avg_score FLOAT, 
	avg_quality FLOAT, 
	high_quality_face_count INTEGER NOT NULL, 
	avatar_relative_path VARCHAR(512), 
	avatar_url VARCHAR(512), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_person_task_person UNIQUE (task_id, person_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: photos

CREATE TABLE photos (
	photo_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	source_photo_id VARCHAR(64) NOT NULL, 
	original_filename VARCHAR(512), 
	stored_filename VARCHAR(512), 
	source_hash VARCHAR(128), 
	mime_type VARCHAR(255), 
	width INTEGER, 
	height INTEGER, 
	taken_at VARCHAR(64), 
	location_json JSON, 
	exif_json JSON, 
	raw_relative_path VARCHAR(512), 
	display_relative_path VARCHAR(512), 
	boxed_relative_path VARCHAR(512), 
	compressed_relative_path VARCHAR(512), 
	raw_url VARCHAR(512), 
	display_url VARCHAR(512), 
	boxed_url VARCHAR(512), 
	compressed_url VARCHAR(512), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (photo_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: profile_context_revisions

CREATE TABLE profile_context_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	primary_person_id VARCHAR(64), 
	payload_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: profile_revisions

CREATE TABLE profile_revisions (
	id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	profile_revision_id VARCHAR(128) NOT NULL, 
	primary_person_id VARCHAR(64), 
	structured_json JSON, 
	report_markdown TEXT, 
	summary TEXT, 
	consistency_json JSON, 
	debug_json JSON, 
	internal_artifacts_json JSON, 
	field_decisions_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: relationship_dossier_revisions

CREATE TABLE relationship_dossier_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	relationship_id VARCHAR(128) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	person_id VARCHAR(64) NOT NULL, 
	dossier_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_relationship_dossier_revision UNIQUE (task_id, relationship_id, revision_no), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: relationship_revisions

CREATE TABLE relationship_revisions (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	relationship_id VARCHAR(128) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	person_id VARCHAR(64) NOT NULL, 
	relationship_type VARCHAR(128), 
	intimacy_score FLOAT, 
	status VARCHAR(64), 
	confidence FLOAT, 
	reasoning TEXT, 
	evidence_json JSON, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_relationship_revision UNIQUE (task_id, relationship_id, revision_no), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: relationship_roots

CREATE TABLE relationship_roots (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	relationship_id VARCHAR(128) NOT NULL, 
	person_id VARCHAR(64) NOT NULL, 
	current_revision_no INTEGER NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_relationship_root_task_rel UNIQUE (task_id, relationship_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: sessions

CREATE TABLE sessions (
	session_id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	token_hash VARCHAR(128) NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (session_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: task_photo_items

CREATE TABLE task_photo_items (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	photo_id VARCHAR(128) NOT NULL, 
	source_photo_id VARCHAR(64) NOT NULL, 
	upload_order INTEGER NOT NULL, 
	batch_no INTEGER, 
	upload_status VARCHAR(32) NOT NULL, 
	source_hash VARCHAR(128), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_task_photo_item UNIQUE (task_id, photo_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: task_stage_records

CREATE TABLE task_stage_records (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	stage_name VARCHAR(32) NOT NULL, 
	stage_version INTEGER, 
	stage_channel VARCHAR(32), 
	status VARCHAR(32) NOT NULL, 
	summary_json JSON, 
	raw_payload_json JSON, 
	normalized_payload_json JSON, 
	artifact_manifest_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_task_stage_name UNIQUE (task_id, stage_name), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: agent_messages

CREATE TABLE agent_messages (
	id VARCHAR(64) NOT NULL, 
	agent_run_id VARCHAR(64) NOT NULL, 
	role VARCHAR(32), 
	seq_no INTEGER NOT NULL, 
	content_text TEXT, 
	content_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id)
);

-- TABLE: agent_outputs

CREATE TABLE agent_outputs (
	id VARCHAR(64) NOT NULL, 
	agent_run_id VARCHAR(64) NOT NULL, 
	output_kind VARCHAR(128), 
	target_id VARCHAR(255), 
	normalized_json JSON, 
	raw_text TEXT, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id)
);

-- TABLE: agent_tool_calls

CREATE TABLE agent_tool_calls (
	id VARCHAR(64) NOT NULL, 
	agent_run_id VARCHAR(64) NOT NULL, 
	tool_name VARCHAR(128), 
	seq_no INTEGER NOT NULL, 
	args_json JSON, 
	result_json JSON, 
	status VARCHAR(32), 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id)
);

-- TABLE: agent_trace_events

CREATE TABLE agent_trace_events (
	id VARCHAR(64) NOT NULL, 
	agent_run_id VARCHAR(64) NOT NULL, 
	event_type VARCHAR(128), 
	seq_no INTEGER NOT NULL, 
	visible_text TEXT, 
	payload_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id)
);

-- TABLE: consistency_check_revisions

CREATE TABLE consistency_check_revisions (
	id VARCHAR(64) NOT NULL, 
	profile_revision_id VARCHAR(128) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	check_name VARCHAR(128) NOT NULL, 
	status VARCHAR(32), 
	details_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(profile_revision_id) REFERENCES profile_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: event_detail_units

CREATE TABLE event_detail_units (
	id VARCHAR(64) NOT NULL, 
	event_revision_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	detail_type VARCHAR(64), 
	detail_text TEXT, 
	normalized_text TEXT, 
	source_refs_json JSON, 
	confidence FLOAT, 
	sort_key INTEGER NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_revision_id) REFERENCES event_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: event_participants

CREATE TABLE event_participants (
	id VARCHAR(64) NOT NULL, 
	event_revision_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	person_id VARCHAR(64) NOT NULL, 
	participant_role VARCHAR(64), 
	confidence FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_revision_id) REFERENCES event_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: event_photo_links

CREATE TABLE event_photo_links (
	id VARCHAR(64) NOT NULL, 
	event_revision_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	source_photo_id VARCHAR(64), 
	evidence_type VARCHAR(32), 
	sort_order INTEGER NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_revision_id) REFERENCES event_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: face_observations

CREATE TABLE face_observations (
	face_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	person_id VARCHAR(64), 
	photo_id VARCHAR(128) NOT NULL, 
	source_photo_id VARCHAR(64), 
	source_hash VARCHAR(128), 
	score FLOAT, 
	similarity FLOAT, 
	quality_score FLOAT, 
	faiss_id INTEGER, 
	bbox_json JSON, 
	bbox_xywh_json JSON, 
	kps_json JSON, 
	match_decision VARCHAR(64), 
	match_reason TEXT, 
	pose_yaw FLOAT, 
	pose_pitch FLOAT, 
	pose_roll FLOAT, 
	pose_bucket VARCHAR(64), 
	eye_visibility_ratio FLOAT, 
	crop_relative_path VARCHAR(512), 
	crop_url VARCHAR(512), 
	boxed_relative_path VARCHAR(512), 
	boxed_url VARCHAR(512), 
	face_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (face_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: ground_truth_assertions

CREATE TABLE ground_truth_assertions (
	id VARCHAR(64) NOT NULL, 
	gt_revision_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	subject_type VARCHAR(64), 
	subject_id VARCHAR(255), 
	field_key VARCHAR(255), 
	operation VARCHAR(64), 
	value_json JSON, 
	evidence_json JSON, 
	note TEXT, 
	is_active BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(gt_revision_id) REFERENCES ground_truth_revisions (gt_revision_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- TABLE: group_members

CREATE TABLE group_members (
	id VARCHAR(64) NOT NULL, 
	group_revision_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	person_id VARCHAR(64) NOT NULL, 
	weight FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(group_revision_id) REFERENCES group_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: photo_assets

CREATE TABLE photo_assets (
	id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	variant_type VARCHAR(32) NOT NULL, 
	relative_path VARCHAR(512), 
	asset_url VARCHAR(512), 
	metadata_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_photo_asset_variant UNIQUE (photo_id, variant_type), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: photo_exif

CREATE TABLE photo_exif (
	id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	raw_exif_json JSON, 
	normalized_exif_json JSON, 
	captured_at VARCHAR(64), 
	gps_lat FLOAT, 
	gps_lng FLOAT, 
	camera_make VARCHAR(128), 
	camera_model VARCHAR(128), 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: profile_fact_decisions

CREATE TABLE profile_fact_decisions (
	id VARCHAR(64) NOT NULL, 
	profile_revision_id VARCHAR(128) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	field_key VARCHAR(255) NOT NULL, 
	payload_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(profile_revision_id) REFERENCES profile_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: profile_field_values

CREATE TABLE profile_field_values (
	id VARCHAR(64) NOT NULL, 
	profile_revision_id VARCHAR(128) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	field_key VARCHAR(255) NOT NULL, 
	domain_name VARCHAR(128), 
	batch_name VARCHAR(128), 
	value_json JSON, 
	confidence FLOAT, 
	reasoning TEXT, 
	traceable_evidence_json JSON, 
	null_reason TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_profile_field_task UNIQUE (task_id, field_key), 
	FOREIGN KEY(profile_revision_id) REFERENCES profile_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: relationship_shared_events

CREATE TABLE relationship_shared_events (
	id VARCHAR(64) NOT NULL, 
	relationship_revision_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	event_id VARCHAR(64), 
	date_snapshot VARCHAR(64), 
	narrative_snapshot TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(relationship_revision_id) REFERENCES relationship_revisions (id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id)
);

-- TABLE: user_heads

CREATE TABLE user_heads (
	user_id VARCHAR(64) NOT NULL, 
	active_dataset_id INTEGER, 
	active_task_id VARCHAR(64), 
	active_profile_revision_id VARCHAR(128), 
	active_gt_revision_id VARCHAR(128), 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (user_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(active_dataset_id) REFERENCES datasets (dataset_id), 
	FOREIGN KEY(active_task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(active_profile_revision_id) REFERENCES profile_revisions (id), 
	FOREIGN KEY(active_gt_revision_id) REFERENCES ground_truth_revisions (gt_revision_id)
);

-- TABLE: vlm_observation_revisions

CREATE TABLE vlm_observation_revisions (
	id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	photo_id VARCHAR(128) NOT NULL, 
	source_photo_id VARCHAR(64), 
	summary TEXT, 
	people_json JSON, 
	relations_json JSON, 
	scene_json JSON, 
	event_json JSON, 
	details_json JSON, 
	clues_json JSON, 
	raw_payload_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: face_embeddings

CREATE TABLE face_embeddings (
	id VARCHAR(64) NOT NULL, 
	face_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	faiss_id INTEGER, 
	embedding vector(512), 
	embedding_dim INTEGER, 
	embedding_model VARCHAR(128), 
	embedding_version INTEGER, 
	embedding_hash VARCHAR(64), 
	source_backend VARCHAR(32), 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_face_embedding_face UNIQUE (face_id), 
	FOREIGN KEY(face_id) REFERENCES face_observations (face_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id)
);

-- TABLE: person_face_links

CREATE TABLE person_face_links (
	id VARCHAR(64) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	dataset_id INTEGER, 
	person_id VARCHAR(64) NOT NULL, 
	face_id VARCHAR(128) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	confidence FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_person_face_link UNIQUE (task_id, person_id, face_id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(dataset_id) REFERENCES datasets (dataset_id), 
	FOREIGN KEY(face_id) REFERENCES face_observations (face_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: vlm_observation_clues

CREATE TABLE vlm_observation_clues (
	id VARCHAR(64) NOT NULL, 
	observation_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	clue_type VARCHAR(64), 
	clue_text TEXT, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(observation_id) REFERENCES vlm_observation_revisions (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: vlm_observation_people

CREATE TABLE vlm_observation_people (
	id VARCHAR(64) NOT NULL, 
	observation_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	person_ref VARCHAR(64), 
	person_id VARCHAR(64), 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(observation_id) REFERENCES vlm_observation_revisions (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- TABLE: vlm_observation_relations

CREATE TABLE vlm_observation_relations (
	id VARCHAR(64) NOT NULL, 
	observation_id VARCHAR(128) NOT NULL, 
	user_id VARCHAR(64) NOT NULL, 
	task_id VARCHAR(64) NOT NULL, 
	photo_id VARCHAR(128) NOT NULL, 
	raw_json JSON, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(observation_id) REFERENCES vlm_observation_revisions (id), 
	FOREIGN KEY(user_id) REFERENCES users (user_id), 
	FOREIGN KEY(task_id) REFERENCES tasks (task_id), 
	FOREIGN KEY(photo_id) REFERENCES photos (photo_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX ix_agent_runs_agent_kind ON agent_runs (agent_kind);
CREATE INDEX ix_agent_runs_task_id ON agent_runs (task_id);
CREATE INDEX ix_agent_runs_user_id ON agent_runs (user_id);
CREATE INDEX ix_artifacts_sha256 ON artifacts (sha256);
CREATE INDEX ix_artifacts_stage ON artifacts (stage);
CREATE INDEX ix_artifacts_storage_backend ON artifacts (storage_backend);
CREATE INDEX ix_artifacts_task_id ON artifacts (task_id);
CREATE INDEX ix_artifacts_user_id ON artifacts (user_id);
CREATE INDEX ix_binary_assets_asset_kind ON binary_assets (asset_kind);
CREATE INDEX ix_binary_assets_dataset_id ON binary_assets (dataset_id);
CREATE INDEX ix_binary_assets_sha256 ON binary_assets (sha256);
CREATE INDEX ix_binary_assets_task_id ON binary_assets (task_id);
CREATE INDEX ix_binary_assets_user_id ON binary_assets (user_id);
CREATE INDEX ix_event_revisions_dataset_id ON event_revisions (dataset_id);
CREATE INDEX ix_event_revisions_date ON event_revisions (date);
CREATE INDEX ix_event_revisions_event_id ON event_revisions (event_id);
CREATE INDEX ix_event_revisions_task_id ON event_revisions (task_id);
CREATE INDEX ix_event_revisions_user_id ON event_revisions (user_id);
CREATE INDEX ix_event_roots_dataset_id ON event_roots (dataset_id);
CREATE INDEX ix_event_roots_event_id ON event_roots (event_id);
CREATE INDEX ix_event_roots_task_id ON event_roots (task_id);
CREATE INDEX ix_event_roots_user_id ON event_roots (user_id);
CREATE INDEX ix_face_recognition_image_policies_is_abandoned ON face_recognition_image_policies (is_abandoned);
CREATE INDEX ix_face_recognition_image_policies_last_image_id ON face_recognition_image_policies (last_image_id);
CREATE INDEX ix_face_recognition_image_policies_last_task_id ON face_recognition_image_policies (last_task_id);
CREATE INDEX ix_face_recognition_image_policies_source_hash ON face_recognition_image_policies (source_hash);
CREATE INDEX ix_face_recognition_image_policies_user_id ON face_recognition_image_policies (user_id);
CREATE INDEX ix_face_reviews_face_id ON face_reviews (face_id);
CREATE INDEX ix_face_reviews_image_id ON face_reviews (image_id);
CREATE INDEX ix_face_reviews_is_inaccurate ON face_reviews (is_inaccurate);
CREATE INDEX ix_face_reviews_person_id ON face_reviews (person_id);
CREATE INDEX ix_face_reviews_source_hash ON face_reviews (source_hash);
CREATE INDEX ix_face_reviews_task_id ON face_reviews (task_id);
CREATE INDEX ix_face_reviews_user_id ON face_reviews (user_id);
CREATE UNIQUE INDEX ix_ground_truth_revisions_gt_revision_id ON ground_truth_revisions (gt_revision_id);
CREATE INDEX ix_ground_truth_revisions_user_id ON ground_truth_revisions (user_id);
CREATE INDEX ix_group_revisions_dataset_id ON group_revisions (dataset_id);
CREATE INDEX ix_group_revisions_group_id ON group_revisions (group_id);
CREATE INDEX ix_group_revisions_task_id ON group_revisions (task_id);
CREATE INDEX ix_group_revisions_user_id ON group_revisions (user_id);
CREATE INDEX ix_group_roots_dataset_id ON group_roots (dataset_id);
CREATE INDEX ix_group_roots_group_id ON group_roots (group_id);
CREATE INDEX ix_group_roots_task_id ON group_roots (task_id);
CREATE INDEX ix_group_roots_user_id ON group_roots (user_id);
CREATE INDEX ix_judge_decision_revisions_dataset_id ON judge_decision_revisions (dataset_id);
CREATE INDEX ix_judge_decision_revisions_judge_type ON judge_decision_revisions (judge_type);
CREATE INDEX ix_judge_decision_revisions_target_id ON judge_decision_revisions (target_id);
CREATE INDEX ix_judge_decision_revisions_task_id ON judge_decision_revisions (task_id);
CREATE INDEX ix_judge_decision_revisions_user_id ON judge_decision_revisions (user_id);
CREATE INDEX ix_object_links_from_id ON object_links (from_id);
CREATE INDEX ix_object_links_relation_type ON object_links (relation_type);
CREATE INDEX ix_object_links_to_id ON object_links (to_id);
CREATE INDEX ix_object_links_user_id ON object_links (user_id);
CREATE INDEX ix_object_registry_dataset_id ON object_registry (dataset_id);
CREATE INDEX ix_object_registry_object_type ON object_registry (object_type);
CREATE INDEX ix_object_registry_parent_id ON object_registry (parent_id);
CREATE INDEX ix_object_registry_root_id ON object_registry (root_id);
CREATE INDEX ix_object_registry_semantic_id ON object_registry (semantic_id);
CREATE INDEX ix_object_registry_task_id ON object_registry (task_id);
CREATE INDEX ix_object_registry_user_id ON object_registry (user_id);
CREATE INDEX ix_person_revisions_dataset_id ON person_revisions (dataset_id);
CREATE INDEX ix_person_revisions_person_id ON person_revisions (person_id);
CREATE INDEX ix_person_revisions_task_id ON person_revisions (task_id);
CREATE INDEX ix_person_revisions_user_id ON person_revisions (user_id);
CREATE INDEX ix_persons_dataset_id ON persons (dataset_id);
CREATE INDEX ix_persons_is_primary_person ON persons (is_primary_person);
CREATE INDEX ix_persons_person_id ON persons (person_id);
CREATE INDEX ix_persons_task_id ON persons (task_id);
CREATE INDEX ix_persons_user_id ON persons (user_id);
CREATE INDEX ix_photos_dataset_id ON photos (dataset_id);
CREATE INDEX ix_photos_source_hash ON photos (source_hash);
CREATE INDEX ix_photos_source_photo_id ON photos (source_photo_id);
CREATE INDEX ix_photos_taken_at ON photos (taken_at);
CREATE INDEX ix_photos_task_id ON photos (task_id);
CREATE INDEX ix_photos_user_id ON photos (user_id);
CREATE INDEX ix_profile_context_revisions_dataset_id ON profile_context_revisions (dataset_id);
CREATE INDEX ix_profile_context_revisions_primary_person_id ON profile_context_revisions (primary_person_id);
CREATE UNIQUE INDEX ix_profile_context_revisions_task_id ON profile_context_revisions (task_id);
CREATE INDEX ix_profile_context_revisions_user_id ON profile_context_revisions (user_id);
CREATE INDEX ix_profile_revisions_dataset_id ON profile_revisions (dataset_id);
CREATE INDEX ix_profile_revisions_primary_person_id ON profile_revisions (primary_person_id);
CREATE INDEX ix_profile_revisions_profile_revision_id ON profile_revisions (profile_revision_id);
CREATE INDEX ix_profile_revisions_task_id ON profile_revisions (task_id);
CREATE INDEX ix_profile_revisions_user_id ON profile_revisions (user_id);
CREATE INDEX ix_relationship_dossier_revisions_dataset_id ON relationship_dossier_revisions (dataset_id);
CREATE INDEX ix_relationship_dossier_revisions_person_id ON relationship_dossier_revisions (person_id);
CREATE INDEX ix_relationship_dossier_revisions_relationship_id ON relationship_dossier_revisions (relationship_id);
CREATE INDEX ix_relationship_dossier_revisions_task_id ON relationship_dossier_revisions (task_id);
CREATE INDEX ix_relationship_dossier_revisions_user_id ON relationship_dossier_revisions (user_id);
CREATE INDEX ix_relationship_revisions_dataset_id ON relationship_revisions (dataset_id);
CREATE INDEX ix_relationship_revisions_person_id ON relationship_revisions (person_id);
CREATE INDEX ix_relationship_revisions_relationship_id ON relationship_revisions (relationship_id);
CREATE INDEX ix_relationship_revisions_task_id ON relationship_revisions (task_id);
CREATE INDEX ix_relationship_revisions_user_id ON relationship_revisions (user_id);
CREATE INDEX ix_relationship_roots_dataset_id ON relationship_roots (dataset_id);
CREATE INDEX ix_relationship_roots_person_id ON relationship_roots (person_id);
CREATE INDEX ix_relationship_roots_relationship_id ON relationship_roots (relationship_id);
CREATE INDEX ix_relationship_roots_task_id ON relationship_roots (task_id);
CREATE INDEX ix_relationship_roots_user_id ON relationship_roots (user_id);
CREATE INDEX ix_sessions_expires_at ON sessions (expires_at);
CREATE UNIQUE INDEX ix_sessions_token_hash ON sessions (token_hash);
CREATE INDEX ix_sessions_user_id ON sessions (user_id);
CREATE INDEX ix_task_photo_items_dataset_id ON task_photo_items (dataset_id);
CREATE INDEX ix_task_photo_items_photo_id ON task_photo_items (photo_id);
CREATE INDEX ix_task_photo_items_source_hash ON task_photo_items (source_hash);
CREATE INDEX ix_task_photo_items_source_photo_id ON task_photo_items (source_photo_id);
CREATE INDEX ix_task_photo_items_task_id ON task_photo_items (task_id);
CREATE INDEX ix_task_photo_items_user_id ON task_photo_items (user_id);
CREATE INDEX ix_task_stage_records_dataset_id ON task_stage_records (dataset_id);
CREATE INDEX ix_task_stage_records_stage_channel ON task_stage_records (stage_channel);
CREATE INDEX ix_task_stage_records_stage_name ON task_stage_records (stage_name);
CREATE INDEX ix_task_stage_records_stage_version ON task_stage_records (stage_version);
CREATE INDEX ix_task_stage_records_task_id ON task_stage_records (task_id);
CREATE INDEX ix_task_stage_records_user_id ON task_stage_records (user_id);
CREATE INDEX ix_agent_messages_agent_run_id ON agent_messages (agent_run_id);
CREATE INDEX ix_agent_outputs_agent_run_id ON agent_outputs (agent_run_id);
CREATE INDEX ix_agent_tool_calls_agent_run_id ON agent_tool_calls (agent_run_id);
CREATE INDEX ix_agent_trace_events_agent_run_id ON agent_trace_events (agent_run_id);
CREATE INDEX ix_consistency_check_revisions_check_name ON consistency_check_revisions (check_name);
CREATE INDEX ix_consistency_check_revisions_profile_revision_id ON consistency_check_revisions (profile_revision_id);
CREATE INDEX ix_consistency_check_revisions_task_id ON consistency_check_revisions (task_id);
CREATE INDEX ix_event_detail_units_detail_type ON event_detail_units (detail_type);
CREATE INDEX ix_event_detail_units_event_revision_id ON event_detail_units (event_revision_id);
CREATE INDEX ix_event_detail_units_task_id ON event_detail_units (task_id);
CREATE INDEX ix_event_participants_event_revision_id ON event_participants (event_revision_id);
CREATE INDEX ix_event_participants_person_id ON event_participants (person_id);
CREATE INDEX ix_event_participants_task_id ON event_participants (task_id);
CREATE INDEX ix_event_photo_links_event_revision_id ON event_photo_links (event_revision_id);
CREATE INDEX ix_event_photo_links_photo_id ON event_photo_links (photo_id);
CREATE INDEX ix_event_photo_links_source_photo_id ON event_photo_links (source_photo_id);
CREATE INDEX ix_event_photo_links_task_id ON event_photo_links (task_id);
CREATE INDEX ix_face_observations_dataset_id ON face_observations (dataset_id);
CREATE INDEX ix_face_observations_faiss_id ON face_observations (faiss_id);
CREATE INDEX ix_face_observations_person_id ON face_observations (person_id);
CREATE INDEX ix_face_observations_photo_id ON face_observations (photo_id);
CREATE INDEX ix_face_observations_source_hash ON face_observations (source_hash);
CREATE INDEX ix_face_observations_source_photo_id ON face_observations (source_photo_id);
CREATE INDEX ix_face_observations_task_id ON face_observations (task_id);
CREATE INDEX ix_face_observations_user_id ON face_observations (user_id);
CREATE INDEX ix_ground_truth_assertions_gt_revision_id ON ground_truth_assertions (gt_revision_id);
CREATE INDEX ix_ground_truth_assertions_subject_id ON ground_truth_assertions (subject_id);
CREATE INDEX ix_ground_truth_assertions_user_id ON ground_truth_assertions (user_id);
CREATE INDEX ix_group_members_group_revision_id ON group_members (group_revision_id);
CREATE INDEX ix_group_members_person_id ON group_members (person_id);
CREATE INDEX ix_group_members_task_id ON group_members (task_id);
CREATE INDEX ix_photo_assets_dataset_id ON photo_assets (dataset_id);
CREATE INDEX ix_photo_assets_photo_id ON photo_assets (photo_id);
CREATE INDEX ix_photo_assets_task_id ON photo_assets (task_id);
CREATE INDEX ix_photo_assets_user_id ON photo_assets (user_id);
CREATE INDEX ix_photo_assets_variant_type ON photo_assets (variant_type);
CREATE INDEX ix_photo_exif_captured_at ON photo_exif (captured_at);
CREATE UNIQUE INDEX ix_photo_exif_photo_id ON photo_exif (photo_id);
CREATE INDEX ix_profile_fact_decisions_field_key ON profile_fact_decisions (field_key);
CREATE INDEX ix_profile_fact_decisions_profile_revision_id ON profile_fact_decisions (profile_revision_id);
CREATE INDEX ix_profile_fact_decisions_task_id ON profile_fact_decisions (task_id);
CREATE INDEX ix_profile_field_values_field_key ON profile_field_values (field_key);
CREATE INDEX ix_profile_field_values_profile_revision_id ON profile_field_values (profile_revision_id);
CREATE INDEX ix_profile_field_values_task_id ON profile_field_values (task_id);
CREATE INDEX ix_relationship_shared_events_event_id ON relationship_shared_events (event_id);
CREATE INDEX ix_relationship_shared_events_relationship_revision_id ON relationship_shared_events (relationship_revision_id);
CREATE INDEX ix_relationship_shared_events_task_id ON relationship_shared_events (task_id);
CREATE INDEX ix_vlm_observation_revisions_dataset_id ON vlm_observation_revisions (dataset_id);
CREATE INDEX ix_vlm_observation_revisions_photo_id ON vlm_observation_revisions (photo_id);
CREATE INDEX ix_vlm_observation_revisions_source_photo_id ON vlm_observation_revisions (source_photo_id);
CREATE INDEX ix_vlm_observation_revisions_task_id ON vlm_observation_revisions (task_id);
CREATE INDEX ix_vlm_observation_revisions_user_id ON vlm_observation_revisions (user_id);
CREATE INDEX ix_face_embeddings_dataset_id ON face_embeddings (dataset_id);
CREATE INDEX ix_face_embeddings_embedding_hash ON face_embeddings (embedding_hash);
CREATE INDEX ix_face_embeddings_face_id ON face_embeddings (face_id);
CREATE INDEX ix_face_embeddings_faiss_id ON face_embeddings (faiss_id);
CREATE INDEX ix_face_embeddings_task_id ON face_embeddings (task_id);
CREATE INDEX ix_face_embeddings_user_id ON face_embeddings (user_id);
CREATE INDEX ix_person_face_links_dataset_id ON person_face_links (dataset_id);
CREATE INDEX ix_person_face_links_face_id ON person_face_links (face_id);
CREATE INDEX ix_person_face_links_person_id ON person_face_links (person_id);
CREATE INDEX ix_person_face_links_photo_id ON person_face_links (photo_id);
CREATE INDEX ix_person_face_links_task_id ON person_face_links (task_id);
CREATE INDEX ix_person_face_links_user_id ON person_face_links (user_id);
CREATE INDEX ix_vlm_observation_clues_clue_type ON vlm_observation_clues (clue_type);
CREATE INDEX ix_vlm_observation_clues_observation_id ON vlm_observation_clues (observation_id);
CREATE INDEX ix_vlm_observation_clues_photo_id ON vlm_observation_clues (photo_id);
CREATE INDEX ix_vlm_observation_clues_task_id ON vlm_observation_clues (task_id);
CREATE INDEX ix_vlm_observation_clues_user_id ON vlm_observation_clues (user_id);
CREATE INDEX ix_vlm_observation_people_observation_id ON vlm_observation_people (observation_id);
CREATE INDEX ix_vlm_observation_people_person_id ON vlm_observation_people (person_id);
CREATE INDEX ix_vlm_observation_people_photo_id ON vlm_observation_people (photo_id);
CREATE INDEX ix_vlm_observation_people_task_id ON vlm_observation_people (task_id);
CREATE INDEX ix_vlm_observation_people_user_id ON vlm_observation_people (user_id);
CREATE INDEX ix_vlm_observation_relations_observation_id ON vlm_observation_relations (observation_id);
CREATE INDEX ix_vlm_observation_relations_photo_id ON vlm_observation_relations (photo_id);
CREATE INDEX ix_vlm_observation_relations_task_id ON vlm_observation_relations (task_id);
CREATE INDEX ix_vlm_observation_relations_user_id ON vlm_observation_relations (user_id);
CREATE INDEX IF NOT EXISTS ix_tasks_user_id ON tasks (user_id);
CREATE INDEX IF NOT EXISTS ix_tasks_dataset_id ON tasks (dataset_id);
CREATE INDEX IF NOT EXISTS ix_tasks_version ON tasks (version);
CREATE INDEX IF NOT EXISTS ix_tasks_pipeline_version ON tasks (pipeline_version);
CREATE INDEX IF NOT EXISTS ix_tasks_pipeline_channel ON tasks (pipeline_channel);
CREATE INDEX IF NOT EXISTS ix_tasks_face_version ON tasks (face_version);
CREATE INDEX IF NOT EXISTS ix_tasks_vlm_version ON tasks (vlm_version);
CREATE INDEX IF NOT EXISTS ix_tasks_lp1_version ON tasks (lp1_version);
CREATE INDEX IF NOT EXISTS ix_tasks_lp2_version ON tasks (lp2_version);
CREATE INDEX IF NOT EXISTS ix_tasks_lp3_version ON tasks (lp3_version);
CREATE INDEX IF NOT EXISTS ix_tasks_judge_version ON tasks (judge_version);
CREATE INDEX IF NOT EXISTS ix_tasks_worker_instance_id ON tasks (worker_instance_id);
CREATE INDEX IF NOT EXISTS ix_tasks_expires_at ON tasks (expires_at);
CREATE INDEX IF NOT EXISTS ix_datasets_user_id ON datasets (user_id);
CREATE INDEX IF NOT EXISTS ix_datasets_dataset_fingerprint ON datasets (dataset_fingerprint);
CREATE INDEX IF NOT EXISTS ix_datasets_first_task_id ON datasets (first_task_id);
CREATE INDEX IF NOT EXISTS ix_datasets_latest_task_id ON datasets (latest_task_id);
CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username);
