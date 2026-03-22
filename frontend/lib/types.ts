export type FaceItem = {
  face_id: string;
  person_id: string;
  score: number;
  similarity: number;
  faiss_id: number;
  bbox: number[];
  image_id: string;
  source_hash?: string | null;
  boxed_image_url?: string | null;
  quality_score?: number;
  quality_flags?: string[];
  match_decision?: string | null;
  match_reason?: string | null;
  pose_yaw?: number | null;
  pose_pitch?: number | null;
  pose_roll?: number | null;
  pose_bucket?: string | null;
  eye_visibility_ratio?: number | null;
  landmark_detected?: boolean;
  landmark_source?: string | null;
  is_inaccurate?: boolean;
  comment_text?: string;
};

export type ImageEntry = {
  image_id: string;
  filename: string;
  source_hash?: string | null;
  timestamp?: string;
  status: string;
  detection_seconds?: number;
  embedding_seconds?: number;
  original_image_url?: string | null;
  display_image_url?: string | null;
  boxed_image_url?: string | null;
  compressed_image_url?: string | null;
  location?: Record<string, unknown>;
  face_count: number;
  faces: FaceItem[];
  failures?: FailureItem[];
  is_abandoned?: boolean;
};

export type PersonGroupImage = {
  image_id: string;
  filename: string;
  timestamp?: string;
  display_image_url?: string | null;
  boxed_image_url?: string | null;
  source_hash?: string | null;
  face_id: string;
  score: number;
  similarity: number;
  quality_score?: number;
  quality_flags?: string[];
  match_decision?: string | null;
  match_reason?: string | null;
  pose_yaw?: number | null;
  pose_pitch?: number | null;
  pose_roll?: number | null;
  pose_bucket?: string | null;
  eye_visibility_ratio?: number | null;
  landmark_detected?: boolean;
  landmark_source?: string | null;
  is_inaccurate?: boolean;
  comment_text?: string;
  is_abandoned?: boolean;
};

export type PersonGroupEntry = {
  person_id: string;
  is_primary?: boolean;
  photo_count: number;
  face_count: number;
  avg_score: number;
  avg_quality?: number;
  high_quality_face_count?: number;
  avatar_url?: string | null;
  images: PersonGroupImage[];
};

export type FailureItem = {
  image_id: string;
  filename: string;
  path: string;
  step: string;
  error: string;
};

export type UploadItem = {
  image_id?: string;
  filename: string;
  stored_filename: string;
  path: string;
  url?: string | null;
  preview_url?: string | null;
  width?: number | null;
  height?: number | null;
  content_type?: string | null;
  source_hash?: string | null;
};

export type FaceRecognitionPayload = {
  primary_person_id?: string | null;
  metrics?: {
    total_images: number;
    total_faces: number;
    total_persons: number;
  };
  persons?: Array<{
    person_id: string;
    photo_count: number;
    face_count: number;
    avg_score: number;
    avg_quality?: number;
    high_quality_face_count?: number;
  }>;
  images: ImageEntry[];
  person_groups: PersonGroupEntry[];
  failed_images: FailureItem[];
};

export type FaceReport = {
  status: string;
  generated_at: string;
  primary_person_id?: string | null;
  total_images: number;
  total_faces: number;
  total_persons: number;
  failed_images: number;
  ambiguous_faces?: number;
  low_quality_faces?: number;
  new_person_from_ambiguity?: number;
  failed_items: Array<{
    image_id: string;
    filename: string;
    step: string;
    error: string;
  }>;
  engine: {
    model_name?: string | null;
    providers?: string[];
  };
  timings: {
    detection_seconds: number;
    embedding_seconds: number;
    total_seconds: number;
    average_image_seconds: number;
  };
  processing: {
    original_uploads_preserved: boolean;
    preview_format: string;
    boxed_format: string;
    recognition_input: string;
  };
  precision_enhancements: string[];
  score_guide: {
    detection_score: string;
    similarity: string;
  };
  no_face_images: Array<{
    image_id: string;
    filename: string;
  }>;
  persons: Array<{
    person_id: string;
    is_primary?: boolean;
    photo_count: number;
    face_count: number;
    avg_score: number;
    avg_quality?: number;
    high_quality_face_count?: number;
  }>;
};

export type TaskEvent = {
  event_id: string;
  date?: string;
  time_range?: string;
  duration?: string;
  title: string;
  type?: string;
  participants?: string[];
  location?: string;
  description?: string;
  photo_count?: number;
  confidence?: number;
  reason?: string;
  narrative?: string;
  narrative_synthesis?: string;
  meta_info?: Record<string, unknown>;
  objective_fact?: Record<string, unknown>;
  social_interaction?: Record<string, unknown>;
  social_dynamics?: Array<Record<string, unknown>>;
  original_image_ids?: string[];
  evidence_photos?: string[];
  lifestyle_tags?: string[];
  tags?: string[];
  social_slices?: Array<Record<string, unknown>>;
  persona_evidence?: Record<string, unknown>;
};

export type TaskRelationship = {
  person_id: string;
  relationship_type: string;
  label: string;
  confidence: number;
  supporting_fact_ids?: string[];
  supporting_original_image_ids?: string[];
  evidence?: Record<string, unknown>;
  reason?: string;
};

export type MemoryProfileField = {
  field_key: string;
  values: string[];
  confidence: number;
  evidence_refs: Array<Record<string, string>>;
  supporting_event_ids: string[];
  supporting_fact_ids?: string[];
  generated_at: string;
  profile_version: number;
  evaluation_status: string;
};

export type MaterializedProfileEntry = {
  value?: string | string[] | null;
  summary?: string | null;
  confidence: number;
  supporting_event_ids?: string[];
  supporting_fact_ids?: string[];
  supporting_photo_ids?: string[];
  evidence_refs: Array<Record<string, string>>;
  updated_at?: string | null;
};

export type MemorySummary = {
  ingestion_id: string;
  profile_version: number;
  photo_count: number;
  person_count: number;
  burst_count: number;
  event_count: number;
  movement_count: number;
  fact_count: number;
  observation_count?: number;
  claim_count?: number;
  relationship_count: number;
  profile_field_count: number;
  segment_count: number;
  external_sinks_published?: number;
  generated_at: string;
};

export type ExternalPublishStatus = {
  status: string;
  reason?: string;
  key_count?: number;
  node_count?: number;
  edge_count?: number;
  segment_count?: number;
  collection?: string;
  keys?: string[];
};

export type MemoryStageSummary = {
  [key: string]: unknown;
};

export type MemoryGraphNode = {
  node_id: string;
  label: string;
  node_type: string;
  ring: number;
  is_primary?: boolean;
  metadata?: Record<string, unknown>;
};

export type MemoryGraphEdge = {
  edge_id: string;
  source_id: string;
  target_id: string;
  edge_type: string;
  label?: string;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type MemoryFocusGraph = {
  center_node_id: string;
  primary_face_person_id?: string | null;
  primary_person_uuid?: string | null;
  node_count: number;
  edge_count: number;
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
  mermaid: string;
};

export type MemoryTransparency = {
  face_stage?: {
    total_faces: number;
    total_persons: number;
    primary_face_person_id?: string | null;
    failed_images: number;
  };
  vlm_stage?: {
    processed_photos: number;
    cached_hits: number;
    runtime_seconds?: number;
    representative_photo_count?: number;
    total_input_photos?: number;
    summaries: MemoryStageSummary[];
  };
  segmentation_stage?: {
    burst_count: number;
    event_count: number;
    movement_count: number;
    summaries: MemoryStageSummary[];
  };
  llm_stage?: {
    fact_count: number;
    relationship_hypothesis_count: number;
    profile_evidence_count: number;
    observation_count?: number;
    claim_count?: number;
    profile_delta_count?: number;
    uncertainty_count?: number;
    slice_count?: number;
    runtime_seconds?: number;
    summaries: MemoryStageSummary[];
  };
  neo4j_state?: {
    node_counts: Record<string, number>;
    edge_count: number;
  };
  focus_graph?: MemoryFocusGraph;
  milvus_state?: {
    segment_count: number;
    segment_type_counts: Record<string, number>;
  };
  redis_state?: {
    profile_version: number;
    published_field_count: number;
    materialized_profile_count?: number;
    relationship_count: number;
    recent_event_count: number;
    recent_fact_count: number;
  };
  object_diff?: {
    change_count: number;
    changes: Array<Record<string, unknown>>;
  };
  traces?: Array<Record<string, unknown>>;
  evidence_chains?: Array<Record<string, unknown>>;
  publish_decisions?: Array<Record<string, unknown>>;
};

export type MilvusSegment = {
  segment_uuid: string;
  tenant_id?: string | null;
  user_id: string;
  photo_uuid?: string | null;
  event_uuid?: string | null;
  person_uuid?: string | null;
  session_uuid?: string | null;
  relationship_uuid?: string | null;
  concept_uuid?: string | null;
  segment_type: string;
  text: string;
  started_at?: string | null;
  ended_at?: string | null;
  place_uuid?: string | null;
  location_hint?: string | null;
  sparse_terms?: string[];
  embedding_source?: string;
  importance_score?: number;
  evidence_refs?: Array<Record<string, string>>;
};

export type MemoryPayload = {
  pipeline_family?: string;
  summary: MemorySummary;
  event_revisions?: Array<Record<string, unknown>>;
  delta_event_revisions?: Array<Record<string, unknown>>;
  atomic_evidence?: Array<Record<string, unknown>>;
  delta_atomic_evidence?: Array<Record<string, unknown>>;
  relationship_revisions?: Array<Record<string, unknown>>;
  delta_relationship_revisions?: Array<Record<string, unknown>>;
  period_revisions?: Array<Record<string, unknown>>;
  profile_revision?: Record<string, unknown>;
  delta_profile_revision?: Record<string, unknown>;
  profile_markdown?: string;
  delta_profile_markdown?: string;
  person_appearances?: Array<Record<string, unknown>>;
  reference_media_signals?: Array<Record<string, unknown>>;
  delta_reference_media_signals?: Array<Record<string, unknown>>;
  envelope: Record<string, unknown>;
  storage: {
    identity_maps?: Record<string, unknown>;
    neo4j?: {
      nodes?: Record<string, Array<Record<string, unknown>>>;
      edges?: Array<Record<string, unknown>>;
      focus_graph?: MemoryFocusGraph;
    };
    milvus?: {
      segments?: MilvusSegment[];
    };
    redis?: {
      profile_current?: Record<string, unknown>;
      profile_revision?: Record<string, unknown>;
      reference_media_catalog?: Record<string, unknown>;
      profile_core?: {
        fields?: Record<string, MemoryProfileField>;
        profile_markdown?: string;
        profiles?: Record<string, Record<string, MaterializedProfileEntry>>;
        uncertainty?: Array<Record<string, unknown>>;
      };
      profile_relationships?: {
        items?: Array<Record<string, unknown>>;
      };
      profile_recent_events?: {
        items?: Array<Record<string, unknown>>;
      };
      profile_recent_facts?: {
        items?: Array<Record<string, unknown>>;
      };
      profile_meta?: Record<string, unknown>;
      profile_debug_refs?: Record<string, unknown>;
      [key: string]: unknown;
    };
    materialization_bundle?: Record<string, unknown>;
  };
  transparency: MemoryTransparency;
  evaluation: Record<string, unknown>;
  external_publish?: {
    enabled: boolean;
    generated_at?: string;
    user_id?: string;
    redis?: ExternalPublishStatus;
    neo4j?: ExternalPublishStatus;
    milvus?: ExternalPublishStatus;
    report_path?: string;
  };
  artifacts?: Record<string, string | null>;
};

export type MemoryQueryAnswer = {
  answer_type: string;
  summary: string;
  confidence: number;
  resolved_entities: Array<Record<string, unknown>>;
  resolved_concepts: string[];
  time_window: Record<string, unknown>;
  supporting_events: Array<Record<string, unknown>>;
  supporting_facts: Array<Record<string, unknown>>;
  supporting_relationships: Array<Record<string, unknown>>;
  representative_photo_ids: string[];
  evidence_segment_ids: string[];
  explanation: string;
  uncertainty_flags: string[];
};

export type MemoryQueryDebugTrace = {
  operator_plan: Record<string, unknown>;
  recall_candidates: Array<Record<string, unknown>>;
  dsl: Record<string, unknown>;
  executed_cypher: string;
  evidence_fill: Record<string, unknown>;
};

export type MemoryQueryResponse = {
  request: Record<string, unknown>;
  answer: MemoryQueryAnswer;
  debug_trace: MemoryQueryDebugTrace;
};

export type MemoryQueryHistoryItem = {
  query_id: string;
  question: string;
  requested_at: string;
  response: MemoryQueryResponse;
};

export type TaskResult = {
  task_id: string;
  generated_at: string;
  summary: {
    title?: string | null;
    total_uploaded: number;
    loaded_images: number;
    failed_images: number;
    face_processed_images: number;
    vlm_processed_images: number;
    total_faces: number;
    total_persons: number;
    primary_person_id?: string | null;
    event_count?: number;
    fact_count?: number;
    relationship_count?: number;
    profile_version?: number;
  };
  face_recognition: FaceRecognitionPayload;
  face_report?: FaceReport | null;
  failed_images: FailureItem[];
  warnings: Array<{ stage: string; message: string }>;
  facts?: TaskEvent[];
  relationships?: TaskRelationship[];
  profile_markdown?: string | null;
  memory_contract?: Record<string, unknown>;
  llm_chunk_artifacts?: Record<string, unknown>;
  dedupe_report?: Record<string, unknown>;
  memory?: MemoryPayload | null;
  artifacts?: Record<string, string | null>;
};

export type TaskState = {
  task_id: string;
  user_id?: string | null;
  version?: string | null;
  status: "draft" | "uploading" | "queued" | "running" | "completed" | "failed";
  stage: string;
  created_at: string;
  updated_at: string;
  upload_count: number;
  uploads?: UploadItem[];
  progress?: TaskProgressState | Record<string, unknown>;
  result?: TaskResult | null;
  result_summary?: Record<string, unknown> | null;
  error?: string | null;
};

export type TaskProgressLogEntry = {
  timestamp: string;
  level: string;
  stage: string;
  substage?: string | null;
  message: string;
  percent?: number | null;
  processed?: number | null;
  total?: number | null;
  provider?: string | null;
  model?: string | null;
  current_person_id?: string | null;
  error?: string | null;
};

export type TaskProgressState = {
  current_stage?: string;
  updated_at?: string;
  stages?: Record<string, unknown>;
  logs?: TaskProgressLogEntry[];
};

export type TaskListResponse = {
  tasks: TaskState[];
};

export type AuthUser = {
  user_id: string;
  username: string;
  created_at: string;
};

export type AuthResponse = {
  user: AuthUser;
};

export type HealthResponse = {
  status: string;
  app_version: string;
  default_task_version: string;
  available_task_versions: string[];
  frontend_origin: string;
  max_upload_photos: number;
  self_registration_enabled?: boolean;
  high_security_mode?: boolean;
  asset_url_prefix: string;
  object_storage_enabled: boolean;
  object_storage_bucket?: string | null;
};
