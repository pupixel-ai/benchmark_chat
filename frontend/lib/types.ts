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
  };
  face_recognition: FaceRecognitionPayload;
  face_report?: FaceReport | null;
  failed_images: FailureItem[];
  warnings: Array<{ stage: string; message: string }>;
  events?: Array<{ title?: string }>;
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
  progress?: Record<string, unknown>;
  result?: TaskResult | null;
  result_summary?: Record<string, unknown> | null;
  error?: string | null;
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
