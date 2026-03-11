export type FaceItem = {
  face_id: string;
  person_id: string;
  score: number;
  similarity: number;
  faiss_id: number;
  bbox: number[];
  image_id: string;
  boxed_image_url?: string | null;
};

export type ImageEntry = {
  image_id: string;
  filename: string;
  timestamp?: string;
  status: string;
  original_image_url?: string | null;
  display_image_url?: string | null;
  boxed_image_url?: string | null;
  compressed_image_url?: string | null;
  location?: Record<string, unknown>;
  face_count: number;
  faces: FaceItem[];
  failures?: FailureItem[];
};

export type FailureItem = {
  image_id: string;
  filename: string;
  path: string;
  step: string;
  error: string;
};

export type UploadItem = {
  filename: string;
  stored_filename: string;
  path: string;
  url?: string | null;
  width?: number | null;
  height?: number | null;
  content_type?: string | null;
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
  }>;
  images: ImageEntry[];
  failed_images: FailureItem[];
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
  failed_images: FailureItem[];
  warnings: Array<{ stage: string; message: string }>;
  events?: Array<{ title?: string }>;
};

export type TaskState = {
  task_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  created_at: string;
  updated_at: string;
  upload_count: number;
  uploads?: UploadItem[];
  progress?: Record<string, unknown>;
  result?: TaskResult | null;
  error?: string | null;
};

export type TaskListResponse = {
  tasks: TaskState[];
};
