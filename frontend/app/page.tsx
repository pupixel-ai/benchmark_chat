"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { AlertTriangle, ArrowUp, Ban, Check, ChevronDown, ChevronRight, Copy, Download, LoaderCircle, MessageSquare, Plus, X } from "lucide-react";
import type {
  AuthResponse,
  AuthUser,
  FaceReport,
  FaceRecognitionPayload,
  FullMemoryEvent,
  FullMemoryPhoto,
  FullMemoryRelationship,
  FullMemoryVlmEntry,
  HealthResponse,
  MemoryQueryHistoryItem,
  MemoryQueryResponse,
  PersonGroupEntry,
  PersonGroupImage,
  TaskMemoryFullResponse,
  TaskMemoryStepsResponse,
  TaskProgressLogEntry,
  TaskListResponse,
  TaskState,
  UploadItem
} from "@/lib/types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_MAX_UPLOADS = 5000;
const MAX_BATCH_FILES = 50;
const MAX_BATCH_BYTES = 64 * 1024 * 1024;
const UPLOAD_BATCH_RETRY_LIMIT = 3;
const UPLOAD_RETRY_BASE_DELAY_MS = 1200;
const GALLERY_PREVIEW_LIMIT = 120;
const FACE_RECOGNITION_STAGES = new Set(["queued", "starting", "loading", "converting", "face_recognition"]);
const FALLBACK_TASK_VERSIONS = ["v0317-Heavy", "v0317", "v0315", "v0312"];
const FALLBACK_DEFAULT_TASK_VERSION = FALLBACK_TASK_VERSIONS[0];
const LEGACY_TASK_VERSION = FALLBACK_TASK_VERSIONS[FALLBACK_TASK_VERSIONS.length - 1];

type PendingUpload = {
  id: string;
  filename: string;
  previewUrl: string;
  sizeLabel: string;
};

type GalleryCard = {
  id: string;
  filename: string;
  imageUrl: string | null;
  meta?: string;
};

type TaskGroup = {
  key: string;
  label: string;
  tasks: TaskState[];
};

const statusLabelMap: Record<TaskState["status"], string> = {
  draft: "草稿",
  uploading: "上传中",
  queued: "排队中",
  running: "处理中",
  completed: "已完成",
  failed: "失败"
};

const stageLabelMap: Record<string, string> = {
  draft: "等待上传",
  uploading: "上传图片",
  queued: "等待执行",
  starting: "初始化",
  loading: "读取图片",
  converting: "格式转换",
  face_recognition: "人脸识别",
  preprocess: "图片压缩",
  vlm: "视觉分析",
  llm: "推理生成",
  memory: "记忆框架",
  completed: "已完成",
  failed: "失败"
};

const llmSubstageLabelMap: Record<string, string> = {
  slice_contract: "Fact Aggregation",
  event_draft: "Event Draft",
  event_merge: "Event Aggregation",
  event_finalize: "Event Finalization",
  global_merge: "Event Aggregation",
  relationship_inference: "Relationship Inference",
  relationship_projector: "Relationship Projection",
  profile_materialization: "Profile Materialization",
  completed: "LLM Completed"
};

const lpStepLabelMap: Record<string, string> = {
  lp1_batch: "LP1 事件聚合",
  lp2_relationship: "LP2 关系推断",
  lp3_profile: "LP3 画像生成"
};

const lpStepStatusLabelMap: Record<string, string> = {
  pending: "等待中",
  running: "进行中",
  completed: "已完成",
  failed: "失败"
};

function toAbsoluteUrl(url?: string | null) {
  if (!url) {
    return null;
  }
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  return `${API_BASE}${url}`;
}

function formatStage(stage: string) {
  return stageLabelMap[stage] ?? stage;
}

function formatLLMSubstage(stage: Record<string, unknown>) {
  const substage = typeof stage.substage === "string" ? stage.substage : "";
  return llmSubstageLabelMap[substage] ?? (substage || "LLM");
}

function formatLpSubstage(substage?: string | null) {
  const key = (substage ?? "").trim();
  return lpStepLabelMap[key] ?? (key || "LP");
}

async function apiFetch(input: string, init?: RequestInit) {
  return fetch(input, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {})
    }
  });
}

function formatStatus(status: TaskState["status"]) {
  return statusLabelMap[status] ?? status;
}

function formatBytes(size: number) {
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatTaskTime(value?: string) {
  if (!value) {
    return "";
  }
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function formatTaskDate(value?: string) {
  if (!value) {
    return "";
  }
  try {
    return new Intl.DateTimeFormat("en-US", {
      month: "2-digit",
      day: "2-digit",
      year: "numeric"
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function formatDateTime(value?: string) {
  if (!value) {
    return "";
  }
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function timestampRank(value?: string | null) {
  if (!value) {
    return 0;
  }
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sortByRecent<T>(items: T[], selector: (item: T) => string | null | undefined) {
  return [...items].sort((left, right) => timestampRank(selector(right)) - timestampRank(selector(left)));
}

function startOfLocalDay(value: Date) {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
}

function taskDayDifference(value?: string) {
  if (!value) {
    return null;
  }
  const target = new Date(value);
  if (Number.isNaN(target.getTime())) {
    return null;
  }
  const todayStart = startOfLocalDay(new Date());
  const targetStart = startOfLocalDay(target);
  return Math.max(0, Math.round((todayStart - targetStart) / (24 * 60 * 60 * 1000)));
}

function taskGroupLabel(value?: string) {
  const difference = taskDayDifference(value);
  if (difference === 0) {
    return "今天";
  }
  if (difference !== null && difference >= 1 && difference <= 3) {
    return `${difference}天前`;
  }
  return formatTaskDate(value);
}

function taskGroupKey(value?: string) {
  const difference = taskDayDifference(value);
  if (difference !== null && difference <= 3) {
    return `recent-${difference}`;
  }
  return `date-${formatTaskDate(value)}`;
}

function taskCreatedAtRank(task: TaskState) {
  const timestamp = Date.parse(task.created_at);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function sortTasksByCreatedAt(tasks: TaskState[]) {
  return [...tasks].sort((left, right) => taskCreatedAtRank(right) - taskCreatedAtRank(left));
}

function groupTasksByCreatedAt(tasks: TaskState[]) {
  const groups = new Map<string, TaskGroup>();
  tasks.forEach((task) => {
    const key = taskGroupKey(task.created_at);
    const existing = groups.get(key);
    if (existing) {
      existing.tasks.push(task);
      return;
    }
    groups.set(key, {
      key,
      label: taskGroupLabel(task.created_at),
      tasks: [task]
    });
  });
  return Array.from(groups.values());
}

function sanitizeTaskSummaryTitle(value?: string | null) {
  if (!value) {
    return null;
  }
  const compact = value.replace(/\s+/g, " ").trim();
  const withoutStructuredSuffix = compact.replace(/\s*@\s*\{.*$/u, "").trim();
  const normalized = withoutStructuredSuffix || compact;
  return normalized.length > 72 ? `${normalized.slice(0, 69)}...` : normalized;
}

function taskSummaryLabel(task: TaskState) {
  const summaryTitle =
    (typeof task.result_summary?.title === "string" ? task.result_summary.title : null) ??
    task.result?.summary?.title ??
    null;
  return sanitizeTaskSummaryTitle(summaryTitle);
}

function taskStableLabel(task: TaskState) {
  return task.task_id.slice(0, 12);
}

function taskDisplayLabel(task: TaskState) {
  return taskSummaryLabel(task) ?? taskStableLabel(task);
}

function toTaskListEntry(task: TaskState): TaskState {
  return {
    ...task,
    result: null,
    uploads: task.uploads,
    progress: task.progress,
  };
}

function uploadMeta(upload: UploadItem) {
  const parts = [];
  if (upload.width && upload.height) {
    parts.push(`${upload.width}×${upload.height}`);
  }
  if (upload.content_type) {
    parts.push(upload.content_type.replace("image/", "").toUpperCase());
  }
  return parts.join(" · ");
}

function buildPendingUploads(files: File[]) {
  return files.slice(0, GALLERY_PREVIEW_LIMIT).map((file, index) => ({
    id: `pending-${index}-${file.name}-${file.size}-${file.lastModified}`,
    filename: file.name,
    previewUrl: URL.createObjectURL(file),
    sizeLabel: formatBytes(file.size)
  }));
}

function revokePendingUploads(items: PendingUpload[]) {
  items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
}

function buildUploadBatches(files: File[]) {
  const batches: File[][] = [];
  let current: File[] = [];
  let currentBytes = 0;

  files.forEach((file) => {
    const nextSize = currentBytes + file.size;
    if (current.length > 0 && (current.length >= MAX_BATCH_FILES || nextSize > MAX_BATCH_BYTES)) {
      batches.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(file);
    currentBytes += file.size;
  });

  if (current.length > 0) {
    batches.push(current);
  }

  return batches;
}

function normalizeFaceReport(faceReport?: FaceReport | null): FaceReport | null {
  if (!faceReport || typeof faceReport !== "object") {
    return null;
  }

  return {
    status: faceReport.status ?? "completed",
    generated_at: faceReport.generated_at ?? "",
    primary_person_id: faceReport.primary_person_id ?? null,
    total_images: Number(faceReport.total_images ?? 0),
    total_faces: Number(faceReport.total_faces ?? 0),
    total_persons: Number(faceReport.total_persons ?? 0),
    failed_images: Number(faceReport.failed_images ?? 0),
    ambiguous_faces: Number(faceReport.ambiguous_faces ?? 0),
    low_quality_faces: Number(faceReport.low_quality_faces ?? 0),
    new_person_from_ambiguity: Number(faceReport.new_person_from_ambiguity ?? 0),
    failed_items: Array.isArray(faceReport.failed_items) ? faceReport.failed_items : [],
    engine: {
      model_name: faceReport.engine?.model_name ?? null,
      providers: Array.isArray(faceReport.engine?.providers) ? faceReport.engine.providers : []
    },
    timings: {
      detection_seconds: Number(faceReport.timings?.detection_seconds ?? 0),
      embedding_seconds: Number(faceReport.timings?.embedding_seconds ?? 0),
      total_seconds: Number(faceReport.timings?.total_seconds ?? 0),
      average_image_seconds: Number(faceReport.timings?.average_image_seconds ?? 0)
    },
    processing: {
      original_uploads_preserved: Boolean(faceReport.processing?.original_uploads_preserved),
      preview_format: faceReport.processing?.preview_format ?? "unknown",
      boxed_format: faceReport.processing?.boxed_format ?? "unknown",
      recognition_input: faceReport.processing?.recognition_input ?? "当前任务未返回识别输入说明"
    },
    precision_enhancements: Array.isArray(faceReport.precision_enhancements)
      ? faceReport.precision_enhancements
      : [],
    score_guide: {
      detection_score: faceReport.score_guide?.detection_score ?? "当前任务未返回检测分数说明",
      similarity: faceReport.score_guide?.similarity ?? "当前任务未返回相似度说明"
    },
    no_face_images: Array.isArray(faceReport.no_face_images) ? faceReport.no_face_images : [],
    persons: Array.isArray(faceReport.persons) ? faceReport.persons : []
  };
}

function WaitingDots({
  label = "处理中",
  compact = false,
  muted = false,
  percent = null
}: {
  label?: string;
  compact?: boolean;
  muted?: boolean;
  percent?: number | null;
}) {
  const tone = muted
    ? "border-black/10 bg-black/5 text-black/40"
    : "border-[#d5c4af] bg-[#f6eee3] text-[#6f5847]";
  const roundedPercent = percent == null ? null : Math.max(0, Math.min(100, Math.round(percent)));

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-[12px] border px-3 py-2 ${
        compact ? "text-xs" : "text-sm"
      } ${tone}`}
    >
      <svg width={compact ? 34 : 44} height={compact ? 12 : 14} viewBox="0 0 44 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" begin="0s" repeatCount="indefinite" />
        </circle>
        <circle cx="22" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" begin="0.18s" repeatCount="indefinite" />
        </circle>
        <circle cx="37" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" begin="0.36s" repeatCount="indefinite" />
        </circle>
      </svg>
      <span>
        {label}
        {roundedPercent != null ? ` · ${roundedPercent}%` : ""}
      </span>
    </div>
  );
}

function LoginPanel({
  mode,
  username,
  password,
  error,
  busy,
  registrationEnabled,
  onModeChange,
  onUsernameChange,
  onPasswordChange,
  onSubmit
}: {
  mode: "login" | "register";
  username: string;
  password: string;
  error: string | null;
  busy: boolean;
  registrationEnabled: boolean;
  onModeChange: (mode: "login" | "register") => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: () => void;
}) {
  const isRegister = registrationEnabled && mode === "register";

  return (
    <section className="mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-[1180px] items-center px-2 py-6 md:px-4">
      <div className="grid w-full gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[12px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.92)] px-6 py-7 shadow-card">
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Memory Engineering</p>
          <h1 className="mt-4 font-display text-5xl leading-[1.06] tracking-tight text-ink md:text-6xl">
            先登录，再进入你自己的任务空间
          </h1>
          <p className="mt-4 max-w-3xl text-base leading-7 text-black/62">
            这是一个原型环境，但从现在开始我们把任务列表、图片和识别结果按用户隔离。你登录后，只会看到你自己的任务。
          </p>
          <div className="mt-6 grid gap-3 md:grid-cols-3">
            <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-4 text-sm text-black/60">
              每个账号拥有独立任务列表
            </div>
            <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-4 text-sm text-black/60">
              任务图片和识别结果跟随当前会话访问
            </div>
            <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-4 text-sm text-black/60">
              原型阶段仍建议仅上传测试数据
            </div>
          </div>
        </div>

        <div className="rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
          <div className="inline-flex rounded-[12px] border border-[#d8c9b7] bg-[#f5ede3] p-1">
            <button
              type="button"
              onClick={() => onModeChange("login")}
              className={`rounded-[10px] px-4 py-2 text-sm transition ${!isRegister ? "bg-white text-ink shadow-sm" : "text-black/55"}`}
            >
              登录
            </button>
            {registrationEnabled ? (
              <button
                type="button"
                onClick={() => onModeChange("register")}
                className={`rounded-[10px] px-4 py-2 text-sm transition ${isRegister ? "bg-white text-ink shadow-sm" : "text-black/55"}`}
              >
                注册
              </button>
            ) : null}
          </div>
          {!registrationEnabled ? <p className="mt-3 text-sm text-black/48">当前环境已关闭自助注册，请使用现有账号登录。</p> : null}

          <div className="mt-6 space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm text-black/60">用户名</span>
              <input
                value={username}
                onChange={(event) => onUsernameChange(event.target.value)}
                placeholder="至少 3 个字符"
                className="w-full rounded-[12px] border border-[#d8c9b7] bg-white/80 px-4 py-3 text-sm outline-none"
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-black/60">密码</span>
              <input
                type="password"
                value={password}
                onChange={(event) => onPasswordChange(event.target.value)}
                placeholder="至少 8 个字符"
                className="w-full rounded-[12px] border border-[#d8c9b7] bg-white/80 px-4 py-3 text-sm outline-none"
              />
            </label>
          </div>

          {error ? <p className="mt-4 text-sm text-[#8a5637]">{error}</p> : null}

          <button
            type="button"
            onClick={onSubmit}
            disabled={busy}
            className="mt-6 inline-flex w-full items-center justify-center rounded-[12px] bg-[#1f1a15] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#2d251e] disabled:cursor-not-allowed disabled:bg-black/20"
          >
            {busy ? (isRegister ? "正在创建账号..." : "正在登录...") : isRegister ? "注册并进入任务台" : "登录进入任务台"}
          </button>
        </div>
      </div>
    </section>
  );
}

function UploadCarousel({
  items,
  showRecognitionBadge,
  totalCount
}: {
  items: GalleryCard[];
  showRecognitionBadge: boolean;
  totalCount: number;
}) {
  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.9)] p-6 shadow-card">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">任务图片走廊</p>
          <p className="mt-2 text-sm leading-6 text-black/60">这里保留刚上传或已入库的图片缩略图、名称和格式信息，方便测试同学快速核对任务内容。</p>
        </div>
        <div className="text-sm text-black/45">
          {totalCount > items.length ? `当前仅渲染前 ${items.length} / ${totalCount} 张缩略图` : "横向滚动查看全部图片"}
        </div>
      </div>

      <div className="mt-5 flex snap-x gap-4 overflow-x-auto pb-2">
        {items.map((item) => (
          <article
            key={item.id}
            className="relative min-w-[260px] snap-start overflow-hidden rounded-[12px] border border-[#ddcebb] bg-[#f7f0e6]"
          >
            <div className="relative h-52 overflow-hidden bg-[#ece2d3]">
              {item.imageUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={item.imageUrl}
                  alt={item.filename}
                  className="h-full w-full object-cover transition duration-300 hover:scale-[1.04]"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-black/35">暂无可预览图片</div>
              )}

              {showRecognitionBadge ? (
                <div className="absolute bottom-3 left-3">
                  <WaitingDots compact label="人脸识别进行中" />
                </div>
              ) : null}
            </div>

            <div className="space-y-2 px-4 py-4">
              <p className="truncate text-base font-medium text-ink">{item.filename}</p>
              {item.meta ? <p className="font-mono text-xs uppercase tracking-[0.14em] text-black/42">{item.meta}</p> : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function formatJson(value: unknown) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

async function copyTextToClipboard(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function readNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function formatRuntimeLabel(value: unknown) {
  const runtime = readNumericValue(value);
  if (runtime == null || runtime <= 0) {
    return "";
  }
  return `${runtime.toFixed(runtime >= 10 ? 1 : 2)}s`;
}

function formatElapsedSince(value?: string | null) {
  if (!value) {
    return "";
  }
  const startedAt = new Date(value).getTime();
  if (!Number.isFinite(startedAt)) {
    return "";
  }
  const seconds = Math.max(0, (Date.now() - startedAt) / 1000);
  return formatRuntimeLabel(seconds);
}

function hasDisplayValue(value: unknown) {
  if (value == null) {
    return false;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function formatLpStepStatus(status?: string | null) {
  const key = (status ?? "").trim();
  return lpStepStatusLabelMap[key] ?? key;
}

function readStageProgress(progress: TaskState["progress"], stage: string) {
  if (!progress || typeof progress !== "object") {
    return {};
  }
  const rawStages = (progress as Record<string, unknown>).stages;
  if (!rawStages || typeof rawStages !== "object") {
    return {};
  }
  const stagePayload = (rawStages as Record<string, unknown>)[stage];
  if (!stagePayload || typeof stagePayload !== "object") {
    return {};
  }
  return stagePayload as Record<string, unknown>;
}

function inferPipelineStageRank(stage: string) {
  if (FACE_RECOGNITION_STAGES.has(stage) || stage === "preprocess") {
    return 1;
  }
  if (stage === "vlm") {
    return 2;
  }
  if (stage === "llm") {
    return 3;
  }
  if (stage === "memory" || stage === "completed" || stage === "failed") {
    return 4;
  }
  return 0;
}

function readTaskLogs(progress: TaskState["progress"]) {
  if (!progress || typeof progress !== "object") {
    return [] as TaskProgressLogEntry[];
  }
  const rawLogs = (progress as Record<string, unknown>).logs;
  if (!Array.isArray(rawLogs)) {
    return [] as TaskProgressLogEntry[];
  }
  return rawLogs.filter((entry): entry is TaskProgressLogEntry => Boolean(entry && typeof entry === "object"));
}

function formatProgressLogStage(entry: TaskProgressLogEntry) {
  if (entry.stage === "llm" && entry.substage) {
    return formatLLMSubstage({ substage: entry.substage });
  }
  return formatStage(entry.stage);
}

function formatProgressLogLine(entry: TaskProgressLogEntry) {
  const parts = [entry.message];
  if (typeof entry.processed === "number" && typeof entry.total === "number" && entry.total > 0) {
    parts.push(`${entry.processed}/${entry.total}`);
  }
  if (typeof entry.current_candidate_index === "number" && typeof entry.total === "number" && entry.total > 0) {
    parts.push(`candidate ${entry.current_candidate_index}/${entry.total}`);
  }
  if (typeof entry.percent === "number") {
    parts.push(`${entry.percent}%`);
  }
  if (entry.current_person_id) {
    parts.push(entry.current_person_id);
  }
  if (entry.last_completed_person_id && entry.last_completed_person_id !== entry.current_person_id) {
    parts.push(`last ${entry.last_completed_person_id}`);
  }
  if (entry.call_started_at && !entry.call_finished_at) {
    const elapsed = formatElapsedSince(entry.call_started_at);
    if (elapsed) {
      parts.push(`call ${elapsed}`);
    }
  }
  if (entry.provider) {
    parts.push(entry.model ? `${entry.provider} · ${entry.model}` : entry.provider);
  }
  if (entry.error) {
    parts.push(entry.error);
  }
  return parts.filter(Boolean).join(" · ");
}

function readLatestStageProviderModel(logs: TaskProgressLogEntry[], stage: string) {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const entry = logs[index];
    if (entry.stage !== stage) {
      continue;
    }
    if (entry.model || entry.provider) {
      return {
        provider: entry.provider ?? "",
        model: entry.model ?? "",
      };
    }
  }
  return {
    provider: "",
    model: "",
  };
}

function formatModelBadgeLabel(model?: string | null, provider?: string | null) {
  const raw = (model ?? provider ?? "").trim();
  if (!raw) {
    return "";
  }
  const normalized = raw.includes("/") ? raw.split("/").pop() ?? raw : raw;
  return normalized;
}

function TaskLogPanel({ logs }: { logs: TaskProgressLogEntry[] }) {
  return (
    <div className="rounded-[12px] border border-[#e2d4a8] bg-[#fff8dc]/85 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#8a6a1f]">Task Logs</p>
          <p className="mt-1 text-xs text-[#8d7740]">按任务运行顺序记录阶段日志与 provider 状态。</p>
        </div>
        <div className="rounded-full bg-white/70 px-3 py-1 text-xs text-[#8a6a1f]">{logs.length} entries</div>
      </div>
      {logs.length > 0 ? (
        <div className="mt-3 max-h-[220px] overflow-auto rounded-[10px] border border-[#eadcae] bg-[#fffaf0] px-3 py-2">
          <div className="space-y-2">
            {logs.map((entry, index) => (
              <div key={`${entry.timestamp}-${index}`} className="rounded-[8px] bg-[#fff2bf] px-3 py-2 text-xs leading-6 text-[#73581e]">
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#8d7740]">
                  <span>{formatDateTime(entry.timestamp)}</span>
                  <span className="rounded-full bg-white/70 px-2 py-0.5 font-mono">{formatProgressLogStage(entry)}</span>
                </div>
                <p className="mt-1 break-all">{formatProgressLogLine(entry)}</p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-[#8d7740]">当前任务还没有运行日志。</p>
      )}
    </div>
  );
}

function TaskErrorPanel({
  errors,
  taskError
}: {
  errors: TaskProgressLogEntry[];
  taskError?: string | null;
}) {
  const combined = [
    ...errors.map((entry, index) => ({
      key: `${entry.timestamp}-${index}`,
      timestamp: entry.timestamp,
      stage: formatProgressLogStage(entry),
      message: entry.error || entry.message,
    })),
    ...(taskError
      ? [
          {
            key: "task-error",
            timestamp: null,
            stage: "Task Failure",
            message: taskError,
          },
        ]
      : []),
  ];
  return (
    <div className="rounded-[12px] border border-[#e8c6c6] bg-[#fff1f1]/88 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-[#9b4a4a]">Task Errors</p>
          <p className="mt-1 text-xs text-[#9f6a6a]">这里会显示阶段错误和最终任务失败原因。</p>
        </div>
        <div className="rounded-full bg-white/80 px-3 py-1 text-xs text-[#9b4a4a]">{combined.length} entries</div>
      </div>
      {combined.length > 0 ? (
        <div className="mt-3 max-h-[180px] overflow-auto rounded-[10px] border border-[#efd6d6] bg-[#fff7f7] px-3 py-2">
          <div className="space-y-2">
            {combined.map((entry) => (
              <div key={entry.key} className="rounded-[8px] bg-[#ffe1e1] px-3 py-2 text-xs leading-6 text-[#7e3a3a]">
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#9f6a6a]">
                  {entry.timestamp ? <span>{formatDateTime(entry.timestamp)}</span> : null}
                  <span className="rounded-full bg-white/80 px-2 py-0.5 font-mono">{entry.stage}</span>
                </div>
                <p className="mt-1 break-all">{entry.message}</p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-[#9f6a6a]">当前任务没有显式错误。</p>
      )}
    </div>
  );
}

function ScrollableJsonPanel({
  title,
  value,
  emptyText,
  loading = false,
  meta,
  badge,
  defaultCollapsed = false,
  loadingLabel = "处理中",
  loadingPercent = null
}: {
  title: string;
  value: unknown;
  emptyText: string;
  loading?: boolean;
  meta?: string;
  badge?: ReactNode;
  defaultCollapsed?: boolean;
  loadingLabel?: string;
  loadingPercent?: number | null;
}) {
  const hasValue = hasDisplayValue(value);
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{title}</p>
            {badge}
          </div>
          {meta ? <p className="text-xs text-black/45">{meta}</p> : null}
        </div>
        <div className="flex items-center gap-2">
          {loading ? <WaitingDots label={loadingLabel} compact muted percent={loadingPercent} /> : null}
          <button
            type="button"
            aria-expanded={!collapsed}
            aria-label={`${collapsed ? "展开" : "折叠"} ${title}`}
            onClick={() => setCollapsed((current) => !current)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#ddcebb] bg-white/85 text-black/52 transition hover:bg-[#f6eee3] hover:text-black focus:outline-none focus:ring-2 focus:ring-[#ccb594] focus:ring-offset-2 focus:ring-offset-white/60"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>
      {collapsed ? null : hasValue ? (
        <pre className="mt-3 max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
          {formatJson(value)}
        </pre>
      ) : loading ? (
        <div className="mt-3">
          <WaitingDots label={loadingLabel} percent={loadingPercent} />
        </div>
      ) : (
        <p className="mt-3 text-sm text-black/56">{emptyText}</p>
      )}
    </div>
  );
}

function FoldableStageCard({
  title,
  meta,
  loading = false,
  loadingLabel = "处理中",
  loadingPercent = null,
  badge,
  defaultCollapsed = false,
  children,
}: {
  title: string;
  meta?: string;
  loading?: boolean;
  loadingLabel?: string;
  loadingPercent?: number | null;
  badge?: ReactNode;
  defaultCollapsed?: boolean;
  children: ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{title}</p>
            {badge}
          </div>
          {meta ? <p className="text-xs text-black/45">{meta}</p> : null}
        </div>
        <div className="flex items-center gap-2">
          {loading ? <WaitingDots label={loadingLabel} compact muted percent={loadingPercent} /> : null}
          <button
            type="button"
            aria-expanded={!collapsed}
            aria-label={`${collapsed ? "展开" : "折叠"} ${title}`}
            onClick={() => setCollapsed((current) => !current)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#ddcebb] bg-white/85 text-black/52 transition hover:bg-[#f6eee3] hover:text-black focus:outline-none focus:ring-2 focus:ring-[#ccb594] focus:ring-offset-2 focus:ring-offset-white/60"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>
      {collapsed ? null : <div className="mt-3">{children}</div>}
    </div>
  );
}

function JsonDetails({
  title,
  value,
  defaultOpen = false
}: {
  title: string;
  value: unknown;
  defaultOpen?: boolean;
}) {
  return (
    <details
      open={defaultOpen}
      className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4"
    >
      <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">
        {title}
      </summary>
      <pre className="mt-3 max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
        {formatJson(value)}
      </pre>
    </details>
  );
}

function JsonCopyButton({
  value,
  ariaLabel,
}: {
  value: unknown;
  ariaLabel: string;
}) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      aria-label={ariaLabel}
      title={ariaLabel}
      onClick={async () => {
        try {
          await copyTextToClipboard(formatJson(value));
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        } catch {
          setCopied(false);
        }
      }}
      className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#ddcebb] bg-white/85 text-black/52 transition hover:bg-[#f6eee3] hover:text-black focus:outline-none focus:ring-2 focus:ring-[#ccb594] focus:ring-offset-2 focus:ring-offset-white/60"
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </button>
  );
}

function InferencePipelinePanel({
  task,
  personIndexPanel,
  personIndexReady,
}: {
  task: TaskState;
  personIndexPanel: ReactNode;
  personIndexReady: boolean;
}) {
  const currentStage = (() => {
    if (task.stage === "completed" || task.stage === "failed") {
      return task.stage;
    }
    const value = task.progress && typeof task.progress === "object" ? (task.progress as Record<string, unknown>).current_stage : null;
    return typeof value === "string" && value ? value : task.stage;
  })();
  const faceStage = readStageProgress(task.progress, "face_recognition");
  const vlmStage = readStageProgress(task.progress, "vlm");
  const llmStage = readStageProgress(task.progress, "llm");

  const facePercent = readNumericValue(faceStage.percent);
  const vlmPercent = readNumericValue(vlmStage.percent);
  const llmPercent = readNumericValue(llmStage.percent);
  const stageRank = inferPipelineStageRank(currentStage);
  const faceRuntime = formatRuntimeLabel(faceStage.runtime_seconds);
  const vlmRuntime = formatRuntimeLabel(vlmStage.runtime_seconds ?? task.result?.memory?.transparency?.vlm_stage?.runtime_seconds);
  const llmRuntime = formatRuntimeLabel(llmStage.runtime_seconds ?? task.result?.memory?.transparency?.llm_stage?.runtime_seconds);
  const taskLogs = readTaskLogs(task.progress);
  const taskErrorLogs = taskLogs.filter((entry) => entry.level === "error" || Boolean(entry.error));
  const llmSubstageLabel = formatLLMSubstage(llmStage);
  const llmProcessedCandidates = readNumericValue(llmStage.processed_candidates);
  const llmFilteredCount = readNumericValue(llmStage.filtered_count);
  const llmCandidateCount = readNumericValue(llmStage.candidate_count);
  const llmCurrentPersonId = typeof llmStage.current_person_id === "string" ? llmStage.current_person_id : "";
  const stageVlmProvider = typeof vlmStage.provider === "string" ? vlmStage.provider : "";
  const stageVlmModel = typeof vlmStage.model === "string" ? vlmStage.model : "";
  const stageLlmProvider = typeof llmStage.provider === "string" ? llmStage.provider : "";
  const stageLlmModel = typeof llmStage.model === "string" ? llmStage.model : "";
  const latestVlmModelInfo = readLatestStageProviderModel(taskLogs, "vlm");
  const latestLlmModelInfo = readLatestStageProviderModel(taskLogs, "llm");
  const vlmProvider = stageVlmProvider || latestVlmModelInfo.provider;
  const vlmModel = stageVlmModel || latestVlmModelInfo.model;
  const llmProvider = stageLlmProvider || latestLlmModelInfo.provider;
  const llmModel = stageLlmModel || latestLlmModelInfo.model;
  const vlmModelBadge = formatModelBadgeLabel(vlmModel, vlmProvider);
  const llmModelBadge = formatModelBadgeLabel(llmModel, llmProvider);
  const vlmLoading = task.status === "running" && currentStage === "vlm";
  const llmLoading = task.status === "running" && currentStage === "llm";
  const llmHasMeasuredProgress =
    (llmProcessedCandidates != null && llmCandidateCount != null && llmCandidateCount > 0) ||
    (llmPercent != null && llmPercent > 10);
  const llmDisplayPercent = llmHasMeasuredProgress ? llmPercent : null;
  const llmDisplayLabel = llmHasMeasuredProgress ? llmSubstageLabel : "LLM 处理中";
  const llmMetaParts = [llmRuntime ? `运行时间 ${llmRuntime}` : ""];
  if (llmLoading && llmSubstageLabel) {
    llmMetaParts.push(llmSubstageLabel);
  }
  if (llmLoading && llmFilteredCount != null) {
    llmMetaParts.push(`候选 ${llmProcessedCandidates ?? 0}/${llmFilteredCount}${llmCandidateCount != null ? `（总计 ${llmCandidateCount}）` : ""}`);
  }
  if (llmLoading && llmCurrentPersonId) {
    llmMetaParts.push(`当前 ${llmCurrentPersonId}`);
  }
  if (llmLoading && llmProvider) {
    llmMetaParts.push(`${llmProvider}${llmModel ? ` · ${llmModel}` : ""}`);
  }
  const llmMeta = llmMetaParts.filter(Boolean).join(" · ");

  const faceValue = task.result?.face_recognition ?? faceStage.face_result_preview ?? null;
  const vlmValue =
    (vlmStage.vlm_results_preview as unknown) ??
    task.result?.memory?.transparency?.vlm_stage?.summaries ??
    null;
  const llmValue =
    (llmStage.memory_contract_preview as unknown) ??
    task.result?.memory?.transparency?.llm_stage?.summaries ??
    (task.result?.memory
      ? {
          event_revisions: task.result.memory.delta_event_revisions ?? task.result.memory.event_revisions ?? [],
          atomic_evidence: task.result.memory.delta_atomic_evidence ?? task.result.memory.atomic_evidence ?? [],
          relationship_revisions:
            task.result.memory.delta_relationship_revisions ?? task.result.memory.relationship_revisions ?? [],
        }
      : null) ??
    null;
  const profileReportValue =
    task.result?.memory?.delta_profile_revision ??
    task.result?.memory?.profile_revision ??
    null;
  const profileMarkdownValue =
    task.result?.memory?.delta_profile_markdown ??
    task.result?.memory?.profile_markdown ??
    task.result?.profile_markdown ??
    (llmStage.profile_markdown_preview as unknown) ??
    null;
  const faceReady = hasDisplayValue(faceValue);
  const vlmReady = hasDisplayValue(vlmValue);
  const llmReady = hasDisplayValue(llmValue);
  const profileReportReady = hasDisplayValue(profileReportValue);
  const profileMarkdownReady = hasDisplayValue(profileMarkdownValue);
  const faceVisible = stageRank >= 1 || faceReady;
  const personIndexVisible = faceVisible;
  const personIndexLoading =
    personIndexVisible &&
    !personIndexReady &&
    (task.status === "running" || task.status === "queued" || task.status === "uploading");
  const vlmVisible = stageRank >= 2 || personIndexReady || vlmReady;
  const llmVisible = stageRank >= 3 || vlmReady || llmReady || profileReportReady || profileMarkdownReady;
  const profileVisible = stageRank >= 4 || llmReady || profileReportReady || profileMarkdownReady;
  const vlmWaiting = task.status === "running" && personIndexReady && !vlmReady;
  const llmWaiting = task.status === "running" && (vlmReady || currentStage === "llm") && !llmReady;
  const profileWaiting =
    task.status === "running" && (llmReady || currentStage === "llm") && !profileReportReady && !profileMarkdownReady;

  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Inference Timeline</p>
          <h2 className="mt-3 text-2xl font-semibold text-ink">推理生成逐阶段展开</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
            前端会按 `Face → 人物索引 → VLM → LLM → Profile` 顺序展示当前任务的阶段结果，并且只展示已经完成的阶段与当前正在运行的阶段。
          </p>
        </div>
        <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-5 py-4">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">Current Stage</p>
          <p className="mt-2 text-xl font-semibold">{currentStage === "llm" ? llmSubstageLabel : formatStage(currentStage)}</p>
        </div>
      </div>

      <div className="mt-5 space-y-4">
        {faceVisible ? (
          <ScrollableJsonPanel
            title="Face Results / 人脸结果"
            value={faceValue}
            emptyText="当前还没有可展示的人脸识别结果。"
            meta={faceRuntime ? `运行时间 ${faceRuntime}` : undefined}
            loading={task.status === "running" && FACE_RECOGNITION_STAGES.has(currentStage) && !hasDisplayValue(faceValue)}
            loadingLabel="人脸识别进行中"
            loadingPercent={facePercent}
          />
        ) : null}

        {personIndexVisible ? (
          <FoldableStageCard
            title="Person Index / 人物索引"
            meta="Face Results ready 后立即展示，随后才继续展示下一步结果。"
            loading={personIndexLoading}
            loadingLabel="人物索引整理中"
          >
            {personIndexPanel}
          </FoldableStageCard>
        ) : null}

        {vlmVisible ? (
          <ScrollableJsonPanel
            title="VLM Results / 视觉分析"
            value={vlmValue}
            emptyText="VLM 结果尚未产出。"
            loading={vlmLoading || vlmWaiting}
            meta={vlmRuntime ? `运行时间 ${vlmRuntime}` : undefined}
            badge={vlmModelBadge ? <span className="inline-flex items-center rounded-full bg-[#e5e5e5] px-3 py-1 text-[11px] font-medium leading-none text-black/55">{vlmModelBadge}</span> : null}
            loadingLabel="VLM 识别进行中"
            loadingPercent={vlmPercent}
          />
        ) : null}

        {llmVisible ? (
          <ScrollableJsonPanel
            title="LLM Results / 推理结果"
            value={llmValue}
            emptyText="LLM 结果尚未产出。"
            loading={llmLoading || llmWaiting}
            meta={llmMeta || undefined}
            badge={llmModelBadge ? <span className="inline-flex items-center rounded-full bg-[#e5e5e5] px-3 py-1 text-[11px] font-medium leading-none text-black/55">{llmModelBadge}</span> : null}
            loadingLabel={llmDisplayLabel}
            loadingPercent={llmDisplayPercent}
          />
        ) : null}

        {profileVisible ? (
          <div className="space-y-4">
            <ScrollableJsonPanel
              title="Profile Report / 用户画像"
              value={profileReportValue}
              emptyText="结构化用户画像尚未产出。"
              loading={profileWaiting}
              meta={llmMeta || undefined}
              badge={llmModelBadge ? <span className="inline-flex items-center rounded-full bg-[#e5e5e5] px-3 py-1 text-[11px] font-medium leading-none text-black/55">{llmModelBadge}</span> : null}
              defaultCollapsed
              loadingLabel="Profile Materialization"
              loadingPercent={llmPercent}
            />
            <ScrollableJsonPanel
              title="Profile Markdown / 画像文本"
              value={profileMarkdownValue}
              emptyText="用户画像 Markdown 尚未产出。"
              loading={profileWaiting}
              meta={llmMeta || undefined}
              loadingLabel="Profile Markdown"
              loadingPercent={llmPercent}
            />
          </div>
        ) : null}

      </div>
      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <TaskLogPanel logs={taskLogs} />
        <TaskErrorPanel errors={taskErrorLogs} taskError={task.error} />
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  detail
}: {
  label: string;
  value: string | number;
  detail?: string;
}) {
  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{label}</p>
      <p className="mt-2 text-xl font-semibold text-ink">{value}</p>
      {detail ? <p className="mt-2 text-sm leading-6 text-black/56">{detail}</p> : null}
    </div>
  );
}

function StageSummaryPanel({
  title,
  items,
  emptyText
}: {
  title: string;
  items: Array<Record<string, unknown>>;
  emptyText: string;
}) {
  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{title}</p>
      <div className="mt-3 space-y-3">
        {items.length > 0 ? (
          items.slice(0, 6).map((item, index) => (
            <div key={`${title}-${index}`} className="rounded-[10px] bg-[#f6eee3] px-3 py-3">
              <pre className="overflow-x-auto whitespace-pre-wrap break-all text-xs leading-6 text-[#5f4e42]">
                {formatJson(item)}
              </pre>
            </div>
          ))
        ) : (
          <p className="text-sm text-black/56">{emptyText}</p>
        )}
      </div>
    </div>
  );
}

function photoPreviewUrl(photo?: FullMemoryPhoto | null) {
  if (!photo) {
    return null;
  }
  return toAbsoluteUrl(photo.display_image_url ?? photo.original_image_url ?? photo.asset_url ?? null);
}

function compactLabel(value: unknown, fallback = "n/a") {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
}

function compactConfidence(value: unknown) {
  const parsed = readNumericValue(value);
  return parsed == null ? "n/a" : parsed.toFixed(2);
}

function PhotoStrip({
  photos,
  emptyText = "当前没有原始照片回挂。"
}: {
  photos: FullMemoryPhoto[];
  emptyText?: string;
}) {
  if (photos.length === 0) {
    return <p className="text-sm text-black/56">{emptyText}</p>;
  }

  return (
    <div className="mt-3 flex gap-3 overflow-x-auto pb-1">
      {photos.map((photo) => {
        const previewUrl = photoPreviewUrl(photo);
        return (
          <article
            key={photo.original_photo_id}
            className="min-w-[156px] overflow-hidden rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed]"
          >
            <a href={previewUrl ?? undefined} target="_blank" rel="noreferrer" className="block">
              <div className="h-28 overflow-hidden bg-[#ece2d3]">
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt={photo.filename ?? photo.original_photo_id} className="h-full w-full object-cover" />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-black/35">暂无预览</div>
                )}
              </div>
            </a>
            <div className="space-y-1 px-3 py-3">
              <p className="truncate text-sm font-medium text-ink">{photo.filename ?? photo.original_photo_id}</p>
              <p className="truncate font-mono text-[11px] uppercase tracking-[0.12em] text-black/42">
                {photo.original_photo_id}
              </p>
              {photo.timestamp ? <p className="text-xs text-black/45">{formatDateTime(photo.timestamp)}</p> : null}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function buildCorePhotoIndex(fullMemory: TaskMemoryFullResponse | null) {
  return new Map<string, FullMemoryPhoto>();
}

function RecallEventCard({ event }: { event: FullMemoryEvent }) {
  return (
    <article className="rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed] p-4">
      <div className="space-y-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Event</p>
          <p className="mt-2 text-sm leading-6 text-[#5f4e42]">{compactLabel(event.llm_summary, "No summary")}</p>
        </div>
        <div className="rounded-[10px] bg-white px-4 py-3 text-xs text-[#6f5847]">
          people {(event.person_ids ?? []).length} · photos {(event.photo_ids ?? []).length} · vlm {(event.vlm ?? []).length}
        </div>
      </div>
      <JsonDetails title="person_ids" value={event.person_ids} />
      <JsonDetails title="photo_ids" value={event.photo_ids} />
      <JsonDetails title="vlm" value={event.vlm} />
    </article>
  );
}

function RecallRelationshipCard({ relationship }: { relationship: FullMemoryRelationship }) {
  return (
    <article className="rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed] p-4">
      <div className="space-y-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Relationship</p>
          <p className="mt-2 text-lg font-semibold text-ink">{compactLabel(relationship.person_id, "Unknown Person")}</p>
        </div>
        <div className="rounded-[10px] bg-white px-4 py-3 text-xs text-[#6f5847]">
          photos {(relationship.photo_ids ?? []).length}
        </div>
      </div>
      <JsonDetails title="photo_ids" value={relationship.photo_ids} />
    </article>
  );
}

function RecallVlmCard({ item }: { item: FullMemoryVlmEntry }) {
  return (
    <article className="rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed] p-4">
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">photo_id</p>
        <p className="mt-2 text-base font-semibold text-ink">{compactLabel(item.photo_id, "photo")}</p>
        <p className="mt-2 text-sm text-[#5f4e42]">persons {(item.person_ids ?? []).length}</p>
      </div>
      <JsonDetails title="person_ids" value={item.person_ids} />
    </article>
  );
}

function QueryResultCard({
  item,
  photoIndex,
}: {
  item: MemoryQueryHistoryItem;
  photoIndex: Map<string, FullMemoryPhoto>;
}) {
  const response = item.response;
  const answer = response.answer;
  const photos = Array.from(
    new Map(
      (answer.original_photo_ids ?? [])
        .map((photoId) => [photoId, photoIndex.get(String(photoId))])
        .filter((entry): entry is [string, FullMemoryPhoto] => Boolean(entry[1]))
    ).values()
  );

  return (
    <article className="rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed] p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{formatDateTime(item.requested_at)}</p>
          <p className="mt-2 text-base font-medium text-ink">{item.question}</p>
          <p className="mt-3 text-sm leading-6 text-[#5f4e42]">{answer.summary}</p>
        </div>
        <div className="space-y-2">
          <div className="rounded-[10px] bg-white px-4 py-3 text-xs text-[#6f5847]">
            {response.query_plan.plan_type} · {answer.answer_type} · confidence {compactConfidence(answer.confidence)}
          </div>
          {response.abstain_reason ? (
            <div className="rounded-[10px] bg-[#fff3f0] px-4 py-3 text-xs text-[#8a5637]">{response.abstain_reason}</div>
          ) : null}
        </div>
      </div>

      <PhotoStrip photos={photos} emptyText="当前回答没有额外原图回挂。" />

      <div className="mt-3 grid gap-3 xl:grid-cols-3">
        <JsonDetails title="Supporting Units" value={response.supporting_units} />
        <JsonDetails title="Supporting Evidence" value={response.supporting_evidence} />
        <JsonDetails title="Supporting Graph Entities" value={response.supporting_graph_entities} />
      </div>
      <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <JsonDetails title="Answer Payload" value={answer} />
        <JsonDetails title="Debug Trace" value={response.debug_trace} />
      </div>
    </article>
  );
}

function MemoryPanel({
  fullMemory,
  loading,
  queryHistory
}: {
  fullMemory: TaskMemoryFullResponse | null;
  loading: boolean;
  queryHistory: MemoryQueryHistoryItem[];
}) {
  const recentQueries = sortByRecent(queryHistory, (item) => item.requested_at);
  const photoIndex = useMemo(() => buildCorePhotoIndex(fullMemory), [fullMemory]);

  if (!fullMemory) {
    return (
      <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Task Memory Core</p>
            <h2 className="mt-3 text-2xl font-semibold text-ink">任务核心记忆</h2>
            <p className="mt-3 text-sm leading-6 text-black/60">
              这里直接展示这个 task 自己的核心结果：用户画像、人物关系、事件和 VLM。
            </p>
          </div>
          {loading ? <WaitingDots label="正在拉取 memory core" /> : null}
        </div>
        {!loading ? <p className="mt-5 text-sm text-black/56">当前任务还没有可展示的 memory core。</p> : null}
      </section>
    );
  }

  const profile = fullMemory.profile;

  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Task Memory Core</p>
          <h2 className="mt-3 text-2xl font-semibold text-ink">任务核心记忆</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
            这里直接展示当前 task 自己的核心产物，不混历史任务，也不展示内部账本。
          </p>
        </div>
        {loading ? <WaitingDots label="正在刷新 memory core" compact /> : null}
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="VLM" value={fullMemory.vlm.length} detail="photo-level understanding" />
        <MetricCard label="Events" value={fullMemory.events.length} detail="task finalized events" />
        <MetricCard label="Relationships" value={fullMemory.relationships.length} detail="task relationship results" />
        <MetricCard label="Profile" value={profile.report_markdown ? "ready" : "pending"} detail="task profile output" />
      </div>

      <div className="mt-5 space-y-4">
        <FoldableStageCard title="Profile / 用户画像">
          {profile.report_markdown ? (
            <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap rounded-[10px] bg-[#f6eee3] p-3 text-sm leading-6 text-[#5f4e42]">
              {profile.report_markdown}
            </pre>
          ) : (
            <p className="text-sm text-black/56">当前画像报告尚未生成。</p>
          )}
        </FoldableStageCard>

        <FoldableStageCard title="Full Relationships / 全量人物关系" meta={`${fullMemory.relationships.length} relationships`}>
          <div className="h-[min(68vh,46rem)] overflow-auto pr-1">
            <div className="space-y-3">
              {fullMemory.relationships.length > 0 ? (
                fullMemory.relationships.map((relationship, index) => (
                  <RecallRelationshipCard key={`${relationship.person_id ?? "relationship"}-${index}`} relationship={relationship} />
                ))
              ) : (
                <p className="text-sm text-black/56">当前任务还没有人物关系结果。</p>
              )}
            </div>
          </div>
        </FoldableStageCard>

        <FoldableStageCard title="Full Events / 全量事件" meta={`${fullMemory.events.length} events`}>
          <div className="h-[min(68vh,46rem)] overflow-auto pr-1">
            <div className="space-y-3">
              {fullMemory.events.length > 0 ? (
                fullMemory.events.map((event, index) => <RecallEventCard key={`event-${index}`} event={event} />)
              ) : (
                <p className="text-sm text-black/56">当前任务还没有全量事件。</p>
              )}
            </div>
          </div>
        </FoldableStageCard>

        <FoldableStageCard title="VLM / 视觉结果" meta={`${fullMemory.vlm.length} items`} defaultCollapsed>
          <div className="space-y-3">
            <p className="text-sm text-black/56">VLM 数量较多时默认折叠，避免把事件和人物关系挤到页面下方。</p>
            {fullMemory.vlm.length > 0 ? (
              fullMemory.vlm.map((item, index) => (
                <RecallVlmCard key={`${item.photo_id ?? index}`} item={item} />
              ))
            ) : (
              <p className="text-sm text-black/56">当前任务还没有 VLM 结果。</p>
            )}
          </div>
        </FoldableStageCard>

        <FoldableStageCard title="Query Recall / 任务召回问答" meta={`${recentQueries.length} questions`}>
          <div className="space-y-3">
            {recentQueries.length > 0 ? (
              recentQueries.map((item) => (
                <QueryResultCard key={item.query_id} item={item} photoIndex={photoIndex} />
              ))
            ) : (
              <p className="text-sm text-black/56">记忆落位后，可以直接在底部 chat bar 里提问；这里会展示新的 query recall 结果。</p>
            )}
          </div>
        </FoldableStageCard>
      </div>
    </section>
  );
}

function MemoryStepsPanel({
  stepsPayload,
  loading,
}: {
  stepsPayload: TaskMemoryStepsResponse | null;
  loading: boolean;
}) {
  const steps = stepsPayload?.steps ?? null;

  const buildMeta = (stepKey: "lp1" | "lp2" | "lp3") => {
    const step = steps?.[stepKey];
    if (!step) {
      return "";
    }
    const summary = (step.summary ?? {}) as Record<string, unknown>;
    const parts = [formatLpStepStatus(step.status)];
    if (typeof step.updated_at === "string" && step.updated_at) {
      parts.push(`更新于 ${formatDateTime(step.updated_at)}`);
    }
    if (stepKey === "lp1") {
      const eventCount = readNumericValue(summary.event_count);
      const batchCount = readNumericValue(summary.batch_count);
      const attemptCount = readNumericValue(summary.attempt_count);
      const retryCount = readNumericValue(summary.retry_count);
      const lastParseStatus = typeof summary.last_parse_status === "string" ? summary.last_parse_status : "";
      if (eventCount != null) {
        parts.push(`${eventCount} events`);
      }
      if (batchCount != null) {
        parts.push(`${batchCount} batches`);
      }
      if (attemptCount != null) {
        parts.push(`${attemptCount} attempts`);
      }
      if (retryCount != null && retryCount > 0) {
        parts.push(`${retryCount} retries`);
      }
      if (lastParseStatus) {
        parts.push(`last ${lastParseStatus}`);
      }
    }
    if (stepKey === "lp2") {
      const relationshipCount = readNumericValue(summary.relationship_count);
      const processedCandidates = readNumericValue(summary.processed_candidates);
      const candidateCount = readNumericValue(summary.candidate_count);
      const currentCandidateIndex = readNumericValue(summary.current_candidate_index);
      const currentPersonId = typeof summary.current_person_id === "string" ? summary.current_person_id : "";
      const lastCompletedPersonId = typeof summary.last_completed_person_id === "string" ? summary.last_completed_person_id : "";
      const callElapsed = formatElapsedSince(typeof summary.call_started_at === "string" ? summary.call_started_at : null);
      if (relationshipCount != null) {
        parts.push(`${relationshipCount} relationships`);
      }
      if (processedCandidates != null && candidateCount != null) {
        parts.push(`${processedCandidates}/${candidateCount} candidates`);
      }
      if (step.status === "running" && currentCandidateIndex != null && candidateCount != null) {
        parts.push(`当前 ${currentCandidateIndex}/${candidateCount}`);
      }
      if (step.status === "running" && currentPersonId) {
        parts.push(`人物 ${currentPersonId}`);
      }
      if (step.status === "running" && callElapsed) {
        parts.push(`本次调用 ${callElapsed}`);
      }
      if (lastCompletedPersonId) {
        parts.push(`上次成功 ${lastCompletedPersonId}`);
      }
    }
    if (stepKey === "lp3") {
      const hasProfile = summary.has_profile === true;
      const reportLength = readNumericValue(summary.report_length);
      if (hasProfile) {
        parts.push("profile ready");
      }
      if (reportLength != null && reportLength > 0) {
        parts.push(`${reportLength} chars`);
      }
    }
    return parts.filter(Boolean).join(" · ");
  };

  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">LP Snapshot Steps</p>
          <h2 className="mt-3 text-2xl font-semibold text-ink">LP1 / LP2 / LP3 分步结果</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
            这里把 `v0323` 的 LP1、LP2、LP3 拆开展示。每一步一旦写出数据，就会立即显示，不再等整条 memory 链路全部结束。
          </p>
        </div>
        {loading ? <WaitingDots label="正在刷新 LP steps" compact /> : null}
      </div>

      <div className="mt-5 space-y-4">
        <FoldableStageCard title="LP1 / 事件聚合" meta={buildMeta("lp1")} loading={loading && !hasDisplayValue(steps?.lp1?.data)} loadingLabel="LP1">
          {hasDisplayValue(steps?.lp1?.data) ? (
            <>
              <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
                {formatJson(steps?.lp1?.data)}
              </pre>
              <div className="mt-3 flex justify-end">
                <JsonCopyButton value={steps?.lp1?.data} ariaLabel="复制 LP1 输出" />
              </div>
            </>
          ) : (
            <p className="text-sm text-black/56">当前还没有 LP1 数据。</p>
          )}
          {hasDisplayValue(steps?.lp1?.attempts) ? <div className="mt-3"><JsonDetails title="LP1 Attempts" value={steps?.lp1?.attempts} /></div> : null}
          {hasDisplayValue(steps?.lp1?.failures) ? <div className="mt-3"><JsonDetails title="LP1 Failures" value={steps?.lp1?.failures} /></div> : null}
        </FoldableStageCard>

        <FoldableStageCard title="LP2 / 关系推断" meta={buildMeta("lp2")} loading={Boolean(steps?.lp2?.status === "running")} loadingLabel="LP2">
          {hasDisplayValue(steps?.lp2?.data) ? (
            <>
              <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
                {formatJson(steps?.lp2?.data)}
              </pre>
              <div className="mt-3 flex justify-end">
                <JsonCopyButton value={steps?.lp2?.data} ariaLabel="复制 LP2 输出" />
              </div>
            </>
          ) : (
            <p className="text-sm text-black/56">当前还没有 LP2 数据。</p>
          )}
          {hasDisplayValue(steps?.lp2?.failures) ? <div className="mt-3"><JsonDetails title="LP2 Failures" value={steps?.lp2?.failures} /></div> : null}
        </FoldableStageCard>

        <FoldableStageCard title="LP3 / 画像生成" meta={buildMeta("lp3")} loading={Boolean(steps?.lp3?.status === "running")} loadingLabel="LP3">
          {hasDisplayValue(steps?.lp3?.data) ? (
            <>
              <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
                {formatJson(steps?.lp3?.data)}
              </pre>
              <div className="mt-3 flex justify-end">
                <JsonCopyButton value={steps?.lp3?.data} ariaLabel="复制 LP3 输出" />
              </div>
            </>
          ) : (
            <p className="text-sm text-black/56">当前还没有 LP3 数据。</p>
          )}
          {hasDisplayValue(steps?.lp3?.failures) ? <div className="mt-3"><JsonDetails title="LP3 Failures" value={steps?.lp3?.failures} /></div> : null}
        </FoldableStageCard>
      </div>
    </section>
  );
}

function PersonGroupsPanel({
  groups,
  commentDrafts,
  expandedComments,
  reviewBusy,
  policyBusy,
  onToggleInaccurate,
  onToggleAbandon,
  onToggleComments,
  onCommentChange,
  onCommentCommit
}: {
  groups: PersonGroupEntry[];
  commentDrafts: Record<string, string>;
  expandedComments: Record<string, boolean>;
  reviewBusy: Record<string, boolean>;
  policyBusy: Record<string, boolean>;
  onToggleInaccurate: (image: PersonGroupImage) => void;
  onToggleAbandon: (image: PersonGroupImage) => void;
  onToggleComments: (faceId: string) => void;
  onCommentChange: (faceId: string, value: string) => void;
  onCommentCommit: (image: PersonGroupImage) => void;
}) {
  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-4 shadow-card">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">人物索引</p>
          <p className="mt-2 text-sm leading-6 text-black/60">每个人用一张代表性人脸和 `Person_ID` 做索引，下面聚合展示所有对应图片。</p>
        </div>
        <div className="text-sm text-black/45">结果区固定高度，超出后滚动浏览</div>
      </div>

      <div className="mt-4 h-[58vh] min-h-[420px] overflow-y-auto pr-1">
        <div className="space-y-4">
          {groups.map((group) => {
            const avatarUrl = toAbsoluteUrl(group.avatar_url);
            return (
              <article
                key={group.person_id}
                className="rounded-[12px] border border-[#ddcebb] bg-[#fbf5ed] p-4"
              >
                <div className="flex flex-col gap-4 md:flex-row md:items-start">
                  <div className="h-20 w-20 shrink-0 overflow-hidden rounded-[12px] border border-[#ddcebb] bg-[#ece2d3]">
                    {avatarUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={avatarUrl} alt={group.person_id} className="h-full w-full object-cover" />
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-black/35">暂无头像</div>
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-lg font-semibold text-ink">{group.person_id}</span>
                      {group.is_primary ? (
                        <span className="rounded-[10px] bg-[#ead8ca] px-2 py-1 font-mono text-xs text-[#8a5637]">
                          主用户
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-2 text-sm text-black/60">
                      对应 {group.photo_count} 张图片 · {group.face_count} 张脸 · 平均检测分数 {group.avg_score.toFixed(3)}
                    </p>
                  </div>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {group.images.map((image) => {
                    const previewUrl = toAbsoluteUrl(image.boxed_image_url ?? image.display_image_url);
                    const commentOpen = Boolean(expandedComments[image.face_id]);
                    const commentValue = commentDrafts[image.face_id] ?? image.comment_text ?? "";
                    const inaccurateActive = Boolean(image.is_inaccurate);
                    const abandonActive = Boolean(image.is_abandoned);
                    return (
                      <article
                        key={`${group.person_id}-${image.image_id}`}
                        className="overflow-hidden rounded-[12px] border border-[#ddcebb] bg-white/70 transition hover:bg-white"
                      >
                        <a href={previewUrl ?? undefined} target="_blank" rel="noreferrer" className="block">
                          <div className="h-40 overflow-hidden bg-[#ece2d3]">
                            {previewUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img src={previewUrl} alt={image.filename} className="h-full w-full object-cover" />
                            ) : (
                              <div className="flex h-full items-center justify-center text-sm text-black/35">暂无图片</div>
                            )}
                          </div>
                        </a>

                        <div className="space-y-2 px-4 py-4">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="truncate text-sm font-medium text-ink">{image.filename}</span>
                            <span className="rounded-[10px] bg-black/5 px-2 py-1 font-mono text-[11px]">
                              {image.image_id}
                            </span>
                          </div>
                          <p className="text-sm text-black/58">
                            分数 {image.score.toFixed(3)} · 相似度 {image.similarity.toFixed(3)}
                          </p>
                          {image.timestamp ? (
                            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-black/40">
                              {formatTaskTime(image.timestamp)}
                            </p>
                          ) : null}

                          <div className="flex items-center justify-end gap-2 pt-1">
                            <button
                              type="button"
                              aria-label="Inaccurate result"
                              title="Inaccurate result"
                              disabled={Boolean(reviewBusy[image.face_id])}
                              onClick={() => onToggleInaccurate(image)}
                              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border transition ${
                                inaccurateActive
                                  ? "border-[#d06f4b] bg-[#fff0e9] text-[#b45631]"
                                  : "border-black/10 bg-[#f6f0e7] text-black/55 hover:bg-white"
                              } disabled:cursor-not-allowed disabled:opacity-50`}
                            >
                              <AlertTriangle size={16} strokeWidth={2.1} />
                            </button>
                            <button
                              type="button"
                              aria-label="Abandon"
                              title="Abandon"
                              disabled={Boolean(policyBusy[image.image_id])}
                              onClick={() => onToggleAbandon(image)}
                              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border transition ${
                                abandonActive
                                  ? "border-[#bb2f2f] bg-[#ffe9e9] text-[#bb2f2f]"
                                  : "border-black/10 bg-[#f6f0e7] text-black/55 hover:bg-white"
                              } disabled:cursor-not-allowed disabled:opacity-50`}
                            >
                              <Ban size={16} strokeWidth={2.1} />
                            </button>
                            <button
                              type="button"
                              aria-label="Comments"
                              title="Comments"
                              disabled={Boolean(reviewBusy[image.face_id])}
                              onClick={() => onToggleComments(image.face_id)}
                              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border transition ${
                                commentOpen || commentValue
                                  ? "border-[#8a5637] bg-[#f6eee3] text-[#8a5637]"
                                  : "border-black/10 bg-[#f6f0e7] text-black/55 hover:bg-white"
                              } disabled:cursor-not-allowed disabled:opacity-50`}
                            >
                              <MessageSquare size={16} strokeWidth={2.1} />
                            </button>
                          </div>

                          {commentOpen ? (
                            <div className="rounded-[12px] border border-[#ddcebb] bg-[#faf4ec] px-3 py-3">
                              <textarea
                                value={commentValue}
                                onChange={(event) => onCommentChange(image.face_id, event.target.value)}
                                onBlur={() => onCommentCommit(image)}
                                placeholder="Comment"
                                className="min-h-[74px] w-full resize-none rounded-[10px] border border-black/8 bg-white/80 px-3 py-2 text-sm text-black/70 outline-none"
                              />
                              <div className="mt-3 flex justify-end">
                                <button
                                  type="button"
                                  aria-label="Submit comment"
                                  title="Submit comment"
                                  disabled={Boolean(reviewBusy[image.face_id])}
                                  onMouseDown={(event) => event.preventDefault()}
                                  onClick={() => onCommentCommit(image)}
                                  className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-[#8a5637] bg-[#8a5637] text-white transition hover:bg-[#6f442c] disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                  <ArrowUp size={16} strokeWidth={2.3} />
                                </button>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function RecallChatDock({
  task,
  hasMemory,
  onQueryComplete
}: {
  task: TaskState | null;
  hasMemory: boolean;
  onQueryComplete: (item: MemoryQueryHistoryItem) => void;
}) {
  const [chatDraft, setChatDraft] = useState("");
  const [isMultiLine, setIsMultiLine] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const canSubmit = Boolean(task && hasMemory && chatDraft.trim() && !isSubmitting);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) {
      return;
    }

    node.style.height = "44px";
    node.style.overflowY = "hidden";
    const nextHeight = Math.min(node.scrollHeight, 72);
    const resolvedHeight = Math.max(44, nextHeight);
    node.style.height = `${resolvedHeight}px`;
    node.style.overflowY = node.scrollHeight > 72 ? "auto" : "hidden";
    setIsMultiLine(resolvedHeight > 44);
  }, [chatDraft]);

  async function submitQuery() {
    if (!task || !hasMemory || !chatDraft.trim()) {
      return;
    }
    setIsSubmitting(true);
    setSubmitError(null);
    const question = chatDraft.trim();

    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${task.task_id}/memory/query`, {
        method: "POST",
        body: JSON.stringify({ question })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "记忆查询失败");
      }
      const payload = (await response.json()) as MemoryQueryResponse;
      onQueryComplete({
        query_id: String(payload.request?.query_id ?? `${task.task_id}-${Date.now()}`),
        question,
        requested_at: new Date().toISOString(),
        response: payload
      });
      setChatDraft("");
    } catch (queryError) {
      setSubmitError(queryError instanceof Error ? queryError.message : "记忆查询失败");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 md:left-[316px]">
      <div className="w-full rounded-none border border-b-0 border-[#d8c9b7] bg-[rgba(248,243,236,0.98)] px-4 pb-3 pt-3 shadow-card backdrop-blur md:px-5">
        {submitError ? (
          <p className="mb-2 text-sm text-[#8a5637]">{submitError}</p>
        ) : null}
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            rows={1}
            value={chatDraft}
            onChange={(event) => setChatDraft(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                void submitQuery();
              }
            }}
            placeholder={
              hasMemory ? "输入记忆召回问题，Cmd/Ctrl + Enter 发送" : "记忆落位完成后可在这里输入召回问题"
            }
            className={`min-h-[44px] flex-1 resize-none border border-black/8 bg-[#f4efe7] px-4 py-[10px] text-sm leading-6 text-black/70 outline-none ${
              isMultiLine ? "rounded-[12px]" : "rounded-full"
            }`}
          />
          <button
            type="button"
            disabled={!canSubmit}
            aria-label="发送"
            onClick={() => void submitQuery()}
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-black/10 bg-[#8a5637] text-white disabled:bg-[#ebe4d8] disabled:text-black/35"
          >
            <ArrowUp size={17} strokeWidth={2.6} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(true);
  const [registrationEnabled, setRegistrationEnabled] = useState(true);
  const [maxUploadPhotos, setMaxUploadPhotos] = useState(DEFAULT_MAX_UPLOADS);
  const [appVersion, setAppVersion] = useState("unknown");
  const [availableTaskVersions, setAvailableTaskVersions] = useState<string[]>(FALLBACK_TASK_VERSIONS);
  const [defaultTaskVersion, setDefaultTaskVersion] = useState(FALLBACK_DEFAULT_TASK_VERSION);
  const [selectedTaskVersion, setSelectedTaskVersion] = useState(FALLBACK_DEFAULT_TASK_VERSION);
  const [normalizeLivePhotos, setNormalizeLivePhotos] = useState(true);
  const [tasks, setTasks] = useState<TaskState[]>([]);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
  const [isDraftView, setIsDraftView] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [commentDrafts, setCommentDrafts] = useState<Record<string, string>>({});
  const [expandedComments, setExpandedComments] = useState<Record<string, boolean>>({});
  const [reviewBusy, setReviewBusy] = useState<Record<string, boolean>>({});
  const [policyBusy, setPolicyBusy] = useState<Record<string, boolean>>({});
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{
    taskId: string;
    totalFiles: number;
    uploadedFiles: number;
    failedFiles: number;
    totalBatches: number;
    completedBatches: number;
    retryAttempt: number;
    lastError?: string | null;
  } | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [isTaskLoading, setIsTaskLoading] = useState(false);
  const [memoryQueryHistoryByTask, setMemoryQueryHistoryByTask] = useState<Record<string, MemoryQueryHistoryItem[]>>({});
  const [fullMemoryByTask, setFullMemoryByTask] = useState<Record<string, TaskMemoryFullResponse | null>>({});
  const [fullMemoryLoadingByTask, setFullMemoryLoadingByTask] = useState<Record<string, boolean>>({});
  const [memoryStepsByTask, setMemoryStepsByTask] = useState<Record<string, TaskMemoryStepsResponse | null>>({});
  const [memoryStepsLoadingByTask, setMemoryStepsLoadingByTask] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const currentFaceStage = currentTask ? readStageProgress(currentTask.progress, "face_recognition") : {};
  const facePreview = (currentFaceStage.face_result_preview as FaceRecognitionPayload | undefined) ?? null;
  const personGroups = currentTask?.result?.face_recognition?.person_groups ?? facePreview?.person_groups ?? [];
  const faceReport = useMemo(() => normalizeFaceReport(currentTask?.result?.face_report ?? null), [currentTask]);
  const memoryResult = currentTask?.result?.memory ?? null;
  const currentUploads = currentTask?.uploads ?? [];
  const taskGroups = useMemo(() => groupTasksByCreatedAt(tasks), [tasks]);
  const currentQueryHistory = currentTask ? memoryQueryHistoryByTask[currentTask.task_id] ?? [] : [];
  const currentFullMemory = currentTask ? fullMemoryByTask[currentTask.task_id] ?? null : null;
  const currentFullMemoryLoading = currentTask ? Boolean(fullMemoryLoadingByTask[currentTask.task_id]) : false;
  const currentMemorySteps = currentTask ? memoryStepsByTask[currentTask.task_id] ?? null : null;
  const currentMemoryStepsLoading = currentTask ? Boolean(memoryStepsLoadingByTask[currentTask.task_id]) : false;
  const currentAnalysisBundle = currentTask?.downloads?.analysis_bundle ?? null;
  const currentTaskVersion = currentTask?.version ?? LEGACY_TASK_VERSION;
  const visibleTaskVersions = useMemo(() => Array.from(new Set([currentTaskVersion, ...availableTaskVersions])), [availableTaskVersions, currentTaskVersion]);
  const currentStatusLabel = currentTask ? formatStatus(currentTask.status) : "";
  const currentStageLabel = currentTask ? formatStage(currentTask.stage) : "";
  const showCurrentStageLabel = Boolean(currentTask && currentStageLabel !== currentStatusLabel);
  const currentTaskSource = currentTask?.options?.creation_source ?? "manual";
  const uploadProgressForCurrentTask =
    currentTask && uploadProgress?.taskId === currentTask.task_id ? uploadProgress : null;
  const persistedUploadTotalCount = Number(currentTask?.options?.expected_upload_count ?? 0);
  const uploadTotalCount = uploadProgressForCurrentTask?.totalFiles ?? persistedUploadTotalCount;
  const uploadUploadedCount = Math.max(
    Number(currentTask?.upload_count ?? 0),
    uploadProgressForCurrentTask?.uploadedFiles ?? 0
  );
  const uploadFailedCount = uploadProgressForCurrentTask?.failedFiles ?? 0;
  const uploadRemainingCount = uploadProgressForCurrentTask
    ? Math.max(uploadTotalCount - uploadUploadedCount - uploadFailedCount, 0)
    : Math.max(uploadTotalCount - uploadUploadedCount, 0);
  const uploadTotalBatches = uploadProgressForCurrentTask?.totalBatches ?? 0;
  const uploadCompletedBatches = uploadProgressForCurrentTask?.completedBatches ?? 0;
  const uploadActiveBatch = uploadTotalBatches > 0 ? Math.min(uploadCompletedBatches + 1, uploadTotalBatches) : 0;
  const uploadRetryAttempt = uploadProgressForCurrentTask?.retryAttempt ?? 0;
  const uploadLastError = uploadProgressForCurrentTask?.lastError ?? null;

  useEffect(() => {
    return () => {
      revokePendingUploads(pendingUploads);
    };
  }, [pendingUploads]);

  const galleryItems = useMemo<GalleryCard[]>(() => {
    if (pendingUploads.length > 0) {
      return pendingUploads.map((item) => ({
        id: item.id,
        filename: item.filename,
        imageUrl: item.previewUrl,
        meta: item.sizeLabel
      }));
    }

    return currentUploads.slice(0, GALLERY_PREVIEW_LIMIT).map((upload) => ({
      id: upload.stored_filename,
      filename: upload.filename,
      imageUrl: toAbsoluteUrl(upload.preview_url ?? upload.url),
      meta: uploadMeta(upload)
    }));
  }, [currentUploads, pendingUploads]);
  const galleryTotalCount = pendingUploads.length > 0 ? selectedFiles.length : currentUploads.length;

  useEffect(() => {
    const nextDrafts: Record<string, string> = {};
    const nextExpanded: Record<string, boolean> = {};
    personGroups.forEach((group) => {
      group.images.forEach((image) => {
        nextDrafts[image.face_id] = image.comment_text ?? "";
        if (image.comment_text) {
          nextExpanded[image.face_id] = true;
        }
      });
    });
    setCommentDrafts(nextDrafts);
    setExpandedComments(nextExpanded);
  }, [personGroups]);

  const showRecognitionBadge =
    pendingUploads.length > 0 ||
    Boolean(
      currentTask &&
        (currentTask.status === "queued" || currentTask.status === "running") &&
        FACE_RECOGNITION_STAGES.has(currentTask.stage)
    );

  async function loadServerConfig() {
    const response = await apiFetch(`${API_BASE}/api/health`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("读取服务配置失败");
    }

    const payload = (await response.json()) as HealthResponse;
    setAppVersion(payload.app_version ?? "unknown");
    setMaxUploadPhotos(Number(payload.max_upload_photos ?? DEFAULT_MAX_UPLOADS));
    const nextVersions =
      Array.isArray(payload.available_task_versions) && payload.available_task_versions.length > 0
        ? payload.available_task_versions
        : FALLBACK_TASK_VERSIONS;
    const nextDefaultVersion = payload.default_task_version ?? nextVersions[0] ?? FALLBACK_DEFAULT_TASK_VERSION;
    setAvailableTaskVersions(nextVersions);
    setDefaultTaskVersion(nextDefaultVersion);
    setSelectedTaskVersion(nextDefaultVersion);
    const enabled = payload.self_registration_enabled !== false;
    setRegistrationEnabled(enabled);
    if (!enabled) {
      setAuthMode("login");
    }
    return payload;
  }

  async function loadCurrentUser() {
    const response = await apiFetch(`${API_BASE}/api/auth/me`, { cache: "no-store" });
    if (response.status === 401) {
      setAuthUser(null);
      setTasks([]);
      setCurrentTask(null);
      setIsDraftView(false);
      return null;
    }
    if (!response.ok) {
      throw new Error("读取登录状态失败");
    }
    const payload = (await response.json()) as AuthResponse;
    setAuthUser(payload.user);
    return payload.user;
  }

  async function submitAuth() {
    setAuthBusy(true);
    setError(null);
    try {
      if (authMode === "register" && !registrationEnabled) {
        throw new Error("注册已关闭");
      }
      const endpoint = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
      const response = await apiFetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: JSON.stringify({
          username: authUsername,
          password: authPassword
        })
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "登录失败");
      }

      const payload = (await response.json()) as AuthResponse;
      setAuthUser(payload.user);
      setAuthPassword("");
      setAuthMode("login");
      await fetchTasks({ selectInitial: true, preserveCurrent: false });
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "登录失败");
    } finally {
      setAuthBusy(false);
    }
  }

  async function logout() {
    setAuthBusy(true);
    try {
      await apiFetch(`${API_BASE}/api/auth/logout`, { method: "POST" });
    } finally {
      setAuthBusy(false);
      setAuthUser(null);
      setTasks([]);
      setCurrentTask(null);
      setIsDraftView(false);
      setSelectedFiles([]);
      setCommentDrafts({});
      setExpandedComments({});
      setReviewBusy({});
      setPolicyBusy({});
      setMemoryQueryHistoryByTask({});
      setFullMemoryByTask({});
      setFullMemoryLoadingByTask({});
      setMemoryStepsByTask({});
      setMemoryStepsLoadingByTask({});
      setUploadProgress(null);
      setPendingUploads((previous) => {
        revokePendingUploads(previous);
        return [];
      });
    }
  }

  async function fetchTasks(options?: { selectInitial?: boolean; preserveCurrent?: boolean }) {
    const selectInitial = options?.selectInitial ?? false;
    const preserveCurrent = options?.preserveCurrent ?? true;

    const response = await apiFetch(`${API_BASE}/api/tasks?limit=30`, { cache: "no-store" });
    if (response.status === 401) {
      setAuthUser(null);
      setTasks([]);
      setCurrentTask(null);
      setIsDraftView(false);
      return;
    }
    if (!response.ok) {
      throw new Error("获取任务列表失败");
    }
    const payload = (await response.json()) as TaskListResponse;
    const orderedTasks = sortTasksByCreatedAt(payload.tasks.map(toTaskListEntry));
    setTasks(orderedTasks);

    if (selectInitial) {
      if (orderedTasks.length > 0) {
        setIsDraftView(false);
        await fetchTask(orderedTasks[0].task_id);
      } else {
        setCurrentTask(null);
        setIsDraftView(true);
      }
      return;
    }

    if (!preserveCurrent || isDraftView) {
      return;
    }

    if (currentTask) {
      const matched = orderedTasks.find((task) => task.task_id === currentTask.task_id);
      if (!matched) {
        setCurrentTask(null);
        setIsDraftView(orderedTasks.length === 0);
      }
    }
  }

  async function fetchTask(taskId: string, options?: { showLoading?: boolean }) {
    const showLoading = options?.showLoading ?? false;
    if (showLoading) {
      setIsTaskLoading(true);
    }
    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${taskId}`, { cache: "no-store" });
      if (response.status === 401) {
        setAuthUser(null);
        setTasks([]);
        setCurrentTask(null);
        setIsDraftView(false);
        return;
      }
      if (!response.ok) {
        throw new Error("读取任务详情失败");
      }
      const payload = (await response.json()) as TaskState;
      setCurrentTask(payload);
      setIsDraftView(false);
      if (payload.version === "v0323" && payload.status !== "draft") {
        void fetchTaskMemorySteps(payload.task_id);
      } else {
        setMemoryStepsByTask((previous) => ({
          ...previous,
          [payload.task_id]: null,
        }));
        setMemoryStepsLoadingByTask((previous) => ({
          ...previous,
          [payload.task_id]: false,
        }));
      }
      if ((payload.result?.memory ?? null) && payload.status === "completed") {
        void fetchTaskFullMemory(payload.task_id);
      } else {
        setFullMemoryByTask((previous) => ({
          ...previous,
          [payload.task_id]: null,
        }));
        setFullMemoryLoadingByTask((previous) => ({
          ...previous,
          [payload.task_id]: false,
        }));
      }
      setTasks((previous) => {
        const taskIndex = previous.findIndex((task) => task.task_id === payload.task_id);
        if (taskIndex === -1) {
          return sortTasksByCreatedAt([...previous, toTaskListEntry(payload)]);
        }

        const next = [...previous];
        next[taskIndex] = toTaskListEntry(payload);
        return next;
      });
      return payload;
    } finally {
      if (showLoading) {
        setIsTaskLoading(false);
      }
    }
  }

  async function fetchTaskMemorySteps(taskId: string) {
    setMemoryStepsLoadingByTask((previous) => ({
      ...previous,
      [taskId]: true,
    }));
    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${taskId}/memory/steps`, { cache: "no-store" });
      if (response.status === 404) {
        setMemoryStepsByTask((previous) => ({
          ...previous,
          [taskId]: null,
        }));
        return;
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "读取 LP steps 失败");
      }
      const payload = (await response.json()) as TaskMemoryStepsResponse;
      setMemoryStepsByTask((previous) => ({
        ...previous,
        [taskId]: payload,
      }));
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "读取 LP steps 失败");
    } finally {
      setMemoryStepsLoadingByTask((previous) => ({
        ...previous,
        [taskId]: false,
      }));
    }
  }

  async function fetchTaskFullMemory(taskId: string) {
    setFullMemoryLoadingByTask((previous) => ({
      ...previous,
      [taskId]: true,
    }));
    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${taskId}/memory/core`, { cache: "no-store" });
      if (response.status === 404) {
        setFullMemoryByTask((previous) => ({
          ...previous,
          [taskId]: null,
        }));
        return;
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "读取 memory core 失败");
      }
      const payload = (await response.json()) as TaskMemoryFullResponse;
      setFullMemoryByTask((previous) => ({
        ...previous,
        [taskId]: payload,
      }));
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "读取 memory core 失败");
    } finally {
      setFullMemoryLoadingByTask((previous) => ({
        ...previous,
        [taskId]: false,
      }));
    }
  }

  useEffect(() => {
    loadServerConfig()
      .then(() => loadCurrentUser())
      .then((user) => {
        if (!user) {
          return;
        }
        return fetchTasks({ selectInitial: true, preserveCurrent: false });
      })
      .catch((loadError) => {
        setError(loadError instanceof Error ? loadError.message : "初始化任务列表失败");
      })
      .finally(() => {
        setAuthBusy(false);
      });
  }, []);

  useEffect(() => {
    if (
      isDraftView ||
      !currentTask ||
      (currentTask.status !== "queued" && currentTask.status !== "running" && currentTask.status !== "uploading")
    ) {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    pollTimerRef.current = window.setInterval(async () => {
      try {
        await fetchTask(currentTask.task_id);
        await fetchTasks();
      } catch {
        // keep polling quietly
      }
    }, 3000);

    return () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [currentTask, isDraftView]);

  async function createTask(files: File[]) {
    setIsUploading(true);
    setError(null);
    let createdTaskId: string | null = null;
    let completedSuccessfully = false;

    try {
      const response = await apiFetch(`${API_BASE}/api/tasks`, {
        method: "POST",
        body: JSON.stringify({
          version: selectedTaskVersion,
          normalize_live_photos: normalizeLivePhotos,
          creation_source: "manual",
          expected_upload_count: files.length,
          requested_max_photos: Math.min(files.length, maxUploadPhotos),
          auto_start_on_upload_complete: true,
        })
      });

      if (response.status === 401) {
        setAuthUser(null);
        throw new Error("登录已失效，请重新登录");
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "创建任务失败");
      }

      const payload = (await response.json()) as { task_id: string };
      createdTaskId = payload.task_id;
      const batches = buildUploadBatches(files);
      let accumulatedFailedCount = 0;
      setUploadProgress({
        taskId: payload.task_id,
        totalFiles: files.length,
        uploadedFiles: 0,
        failedFiles: 0,
        totalBatches: batches.length,
        completedBatches: 0,
        retryAttempt: 0,
        lastError: null,
      });
      setIsDraftView(false);
      await fetchTask(payload.task_id);

      const uploadSingleBatch = async (batch: File[], batchIndex: number) => {
        for (let attempt = 1; attempt <= UPLOAD_BATCH_RETRY_LIMIT; attempt += 1) {
          setUploadProgress((previous) =>
            previous && previous.taskId === payload.task_id
              ? {
                  ...previous,
                  retryAttempt: attempt,
                  lastError: null,
                }
              : previous
          );

          try {
            const formData = new FormData();
            formData.append("creation_source", "manual");
            formData.append("expected_upload_count", String(files.length));
            formData.append("requested_max_photos", String(Math.min(files.length, maxUploadPhotos)));
            formData.append("auto_start_on_upload_complete", "true");
            batch.forEach((file) => formData.append("files", file));

            const batchResponse = await apiFetch(`${API_BASE}/api/tasks/${payload.task_id}/upload-batches`, {
              method: "POST",
              body: formData,
            });
            if (!batchResponse.ok) {
              const batchPayload = await batchResponse.json().catch(() => null);
              throw new Error(batchPayload?.detail ?? `第 ${batchIndex + 1} 批上传失败`);
            }

            const batchPayload = (await batchResponse.json()) as {
              upload_count?: number;
              failed_count?: number;
              status?: TaskState["status"];
              stage?: string;
            };
            accumulatedFailedCount += Number(batchPayload.failed_count ?? 0);
            setUploadProgress((previous) =>
              previous && previous.taskId === payload.task_id
                ? {
                    ...previous,
                    uploadedFiles: Math.max(previous.uploadedFiles, Number(batchPayload.upload_count ?? previous.uploadedFiles)),
                    failedFiles: previous.failedFiles + Number(batchPayload.failed_count ?? 0),
                    completedBatches: Math.max(previous.completedBatches, batchIndex + 1),
                    retryAttempt: 0,
                    lastError: null,
                  }
                : previous
            );
            setCurrentTask((previous) =>
              previous && previous.task_id === payload.task_id
                ? {
                    ...previous,
                    status: batchPayload.status ?? "uploading",
                    stage: batchPayload.stage ?? "uploading",
                    upload_count: Number(batchPayload.upload_count ?? previous.upload_count ?? 0),
                  }
                : previous
            );
            return;
          } catch (batchError) {
            const message = batchError instanceof Error ? batchError.message : `第 ${batchIndex + 1} 批上传失败`;
            setUploadProgress((previous) =>
              previous && previous.taskId === payload.task_id
                ? {
                    ...previous,
                    lastError: `第 ${batchIndex + 1}/${batches.length} 批失败：${message}`,
                  }
                : previous
            );
            if (attempt >= UPLOAD_BATCH_RETRY_LIMIT) {
              throw new Error(`上传在第 ${batchIndex + 1}/${batches.length} 批中断：${message}`);
            }
            await sleep(UPLOAD_RETRY_BASE_DELAY_MS * attempt);
          }
        }
      };

      for (let index = 0; index < batches.length; index += 1) {
        await uploadSingleBatch(batches[index], index);
      }

      const latestTask = await fetchTask(payload.task_id);
      const latestUploadCount = Number(latestTask?.upload_count ?? 0);
      const accountedFileCount = latestUploadCount + accumulatedFailedCount;
      if (accountedFileCount < files.length) {
        throw new Error(`上传未完成：服务端仅确认 ${accountedFileCount} / ${files.length} 张，请稍后重试剩余分片`);
      }

      await fetchTasks();
      setUploadProgress(null);
      setPendingUploads([]);
      completedSuccessfully = true;
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "创建任务失败");
      if (createdTaskId) {
        await fetchTask(createdTaskId).catch(() => null);
      }
    } finally {
      setIsUploading(false);
      if (completedSuccessfully) {
        setSelectedFiles([]);
        if (createdTaskId && fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    }
  }

  async function updateFaceReview(image: PersonGroupImage, next: { is_inaccurate: boolean; comment_text: string }) {
    if (!currentTask) {
      return;
    }
    setReviewBusy((previous) => ({ ...previous, [image.face_id]: true }));
    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${currentTask.task_id}/faces/${image.face_id}/review`, {
        method: "PUT",
        body: JSON.stringify(next)
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "更新人脸反馈失败");
      }
      await fetchTask(currentTask.task_id);
      await fetchTasks();
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "更新人脸反馈失败");
    } finally {
      setReviewBusy((previous) => ({ ...previous, [image.face_id]: false }));
    }
  }

  async function toggleInaccurate(image: PersonGroupImage) {
    const commentText = commentDrafts[image.face_id] ?? image.comment_text ?? "";
    await updateFaceReview(image, {
      is_inaccurate: !Boolean(image.is_inaccurate),
      comment_text: commentText
    });
  }

  async function commitComment(image: PersonGroupImage) {
    const nextComment = commentDrafts[image.face_id] ?? "";
    if (nextComment === (image.comment_text ?? "")) {
      return;
    }
    await updateFaceReview(image, {
      is_inaccurate: Boolean(image.is_inaccurate),
      comment_text: nextComment
    });
  }

  async function toggleAbandon(image: PersonGroupImage) {
    if (!currentTask) {
      return;
    }
    setPolicyBusy((previous) => ({ ...previous, [image.image_id]: true }));
    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${currentTask.task_id}/images/${image.image_id}/face-policy`, {
        method: "PUT",
        body: JSON.stringify({
          is_abandoned: !Boolean(image.is_abandoned)
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "更新图片 abandon 失败");
      }
      await fetchTask(currentTask.task_id);
      await fetchTasks();
    } catch (policyError) {
      setError(policyError instanceof Error ? policyError.message : "更新图片 abandon 失败");
    } finally {
      setPolicyBusy((previous) => ({ ...previous, [image.image_id]: false }));
    }
  }

  function recordMemoryQuery(taskId: string, item: MemoryQueryHistoryItem) {
    setMemoryQueryHistoryByTask((previous) => {
      const nextItems = sortByRecent([item, ...(previous[taskId] ?? [])], (entry) => entry.requested_at).slice(0, 12);
      return {
        ...previous,
        [taskId]: nextItems,
      };
    });
  }

  async function deleteTask(task: TaskState) {
    if (task.status !== "completed" && task.status !== "failed") {
      setError("任务处理中，暂不支持删除");
      return;
    }

    const confirmed = window.confirm(`确认删除任务“${taskDisplayLabel(task)}”及其上传图片、人脸结果和缓存文件吗？此操作不可恢复。`);
    if (!confirmed) {
      return;
    }

    setDeletingTaskId(task.task_id);
    setError(null);

    try {
      const response = await apiFetch(`${API_BASE}/api/tasks/${task.task_id}`, {
        method: "DELETE",
      });

      if (response.status === 401) {
        setAuthUser(null);
        throw new Error("登录已失效，请重新登录");
      }

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "删除任务失败");
      }

      setTasks((previous) => previous.filter((item) => item.task_id !== task.task_id));
      setCurrentTask((previous) => (previous?.task_id === task.task_id ? null : previous));
      setMemoryQueryHistoryByTask((previous) => {
        const next = { ...previous };
        delete next[task.task_id];
        return next;
      });
      setFullMemoryByTask((previous) => {
        const next = { ...previous };
        delete next[task.task_id];
        return next;
      });
      setFullMemoryLoadingByTask((previous) => {
        const next = { ...previous };
        delete next[task.task_id];
        return next;
      });
      setMemoryStepsByTask((previous) => {
        const next = { ...previous };
        delete next[task.task_id];
        return next;
      });
      setMemoryStepsLoadingByTask((previous) => {
        const next = { ...previous };
        delete next[task.task_id];
        return next;
      });
      setIsDraftView((previous) => previous || currentTask?.task_id === task.task_id);
      await fetchTasks({ selectInitial: !currentTask || currentTask.task_id === task.task_id, preserveCurrent: false });
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "删除任务失败");
    } finally {
      setDeletingTaskId(null);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const incomingFiles = Array.from(event.target.files ?? []);
    const exceededLimit = incomingFiles.length > maxUploadPhotos;
    const files = incomingFiles.slice(0, maxUploadPhotos);
    if (files.length === 0) {
      return;
    }

    setError(exceededLimit ? `当前环境最多上传 ${maxUploadPhotos} 张图片，已自动截断` : null);
    setSelectedFiles(files);
    setPendingUploads((previous) => {
      revokePendingUploads(previous);
      return buildPendingUploads(files);
    });
    void createTask(files);
  }

  function openDraftTask() {
    setIsDraftView(true);
    setCurrentTask(null);
    setError(null);
    setSelectedTaskVersion(defaultTaskVersion);
    setSelectedFiles([]);
    setCommentDrafts({});
    setExpandedComments({});
    setReviewBusy({});
    setPolicyBusy({});
    setFullMemoryLoadingByTask({});
    setMemoryStepsLoadingByTask({});
    setUploadProgress(null);
    setPendingUploads((previous) => {
      revokePendingUploads(previous);
      return [];
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  if (authBusy && !authUser) {
    return (
      <main className="min-h-screen px-2 py-6 md:px-4">
        <section className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-[1180px] items-center justify-center">
          <WaitingDots label="正在检查登录状态" />
        </section>
      </main>
    );
  }

  if (!authUser) {
    return (
      <main className="min-h-screen">
        <LoginPanel
          mode={authMode}
          username={authUsername}
          password={authPassword}
          error={error}
          busy={authBusy}
          registrationEnabled={registrationEnabled}
          onModeChange={setAuthMode}
          onUsernameChange={setAuthUsername}
          onPasswordChange={setAuthPassword}
          onSubmit={() => void submitAuth()}
        />
      </main>
    );
  }

  const draftSelected = isDraftView && !currentTask;
  const personIndexPanel =
    personGroups.length > 0 ? (
      <PersonGroupsPanel
        groups={personGroups}
        commentDrafts={commentDrafts}
        expandedComments={expandedComments}
        reviewBusy={reviewBusy}
        policyBusy={policyBusy}
        onToggleInaccurate={(image) => void toggleInaccurate(image)}
        onToggleAbandon={(image) => void toggleAbandon(image)}
        onToggleComments={(faceId) =>
          setExpandedComments((previous) => ({
            ...previous,
            [faceId]: !previous[faceId]
          }))
        }
        onCommentChange={(faceId, value) =>
          setCommentDrafts((previous) => ({
            ...previous,
            [faceId]: value
          }))
        }
        onCommentCommit={(image) => void commitComment(image)}
      />
    ) : currentTask && (currentTask.status === "running" || currentTask.status === "queued" || currentTask.status === "uploading") ? (
      <WaitingDots label="人物索引整理中" />
    ) : faceReport?.total_persons === 0 ? (
      <p className="text-sm text-black/58">静态帧未检测到人脸，因此没有人物索引。</p>
    ) : (
      <p className="text-sm text-black/58">当前还没有可展示的人物索引。</p>
    );
  const personIndexReady = personGroups.length > 0 || Boolean(faceReport) || Boolean(facePreview);

  return (
    <main className="min-h-screen px-2 py-6 md:px-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.heic,.heif,.livp"
        multiple
        onChange={handleFileChange}
        disabled={isUploading}
        className="hidden"
      />

      <div className="mx-auto flex max-w-[1680px] gap-0">
        <aside className="min-h-[calc(100vh-3rem)] w-full max-w-[300px] shrink-0 border-r border-black/10 pr-3">
          <div className="sticky top-6 flex h-[calc(100vh-3rem)] flex-col">
            <div className="space-y-3 px-2.5 pb-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-black/70">任务列表</p>
                <button
                  type="button"
                  onClick={openDraftTask}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-black/10 bg-white/70 text-black/65 transition hover:bg-white"
                  aria-label="新建任务"
                >
                  <Plus size={17} strokeWidth={2.6} />
                </button>
              </div>
              <div className="rounded-[12px] border border-[#ddcebb] bg-white/65 px-3 py-3">
                <p className="truncate text-sm font-medium text-ink">{authUser.username}</p>
                <button
                  type="button"
                  onClick={() => void logout()}
                  className="mt-2 text-xs text-black/45 transition hover:text-black/70"
                >
                  退出登录
                </button>
              </div>
            </div>

            <div className="space-y-4 overflow-y-auto pr-1">
              {draftSelected ? (
                <button
                  type="button"
                  onClick={openDraftTask}
                  className="w-full rounded-[12px] bg-white/75 px-2.5 py-3 text-left shadow-sm"
                >
                  <p className="truncate text-sm font-medium text-ink">新的测试任务</p>
                <p className="mt-1 text-xs text-black/45">等待上传图片</p>
                </button>
              ) : null}

              {taskGroups.map((group) => (
                <section key={group.key} className="space-y-1">
                  <p className="px-2.5 font-mono text-[11px] uppercase tracking-[0.18em] text-black/36">{group.label}</p>
                  {group.tasks.map((task) => {
                    const active = !isDraftView && currentTask?.task_id === task.task_id;
                    const deleting = deletingTaskId === task.task_id;
                    const stableLabel = taskStableLabel(task);
                    const summaryLabel = taskSummaryLabel(task);
                    return (
                      <div
                        key={task.task_id}
                        className={`w-full rounded-[12px] px-2.5 py-3 text-left transition ${
                          active ? "bg-white/75 shadow-sm" : "hover:bg-white/45"
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          <button
                            type="button"
                            onClick={() => fetchTask(task.task_id, { showLoading: true }).catch(() => null)}
                            className="min-w-0 flex-1 text-left"
                          >
                            <p className="truncate font-mono text-[12px] font-semibold tracking-[0.12em] text-ink">
                              {stableLabel}
                            </p>
                            {summaryLabel ? (
                              <p className="mt-1 truncate text-sm font-medium text-ink/88">{summaryLabel}</p>
                            ) : null}
                            <p className="mt-1 truncate text-xs text-black/45">
                              {formatStatus(task.status)} · {formatTaskTime(task.created_at) || formatStage(task.stage)}
                            </p>
                            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-black/45">
                              <span className="truncate">任务版本 {task.version ?? defaultTaskVersion}</span>
                              {task.options?.creation_source === "api" ? (
                                <span className="rounded-full border border-[#d7c3a8] bg-[#f6eee3] px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-[#7a5a37]">
                                  API
                                </span>
                              ) : null}
                            </div>
                          </button>
                          <button
                            type="button"
                            onClick={() => void deleteTask(task)}
                            disabled={deleting || (task.status !== "completed" && task.status !== "failed")}
                            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-black/35 transition hover:bg-black/5 hover:text-black/60 disabled:cursor-not-allowed disabled:text-black/15"
                            aria-label={`删除任务 ${taskDisplayLabel(task)}`}
                            title={task.status === "completed" || task.status === "failed" ? "删除任务及其文件" : "任务处理中，暂不可删除"}
                          >
                            <X size={16} strokeWidth={2.3} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </section>
              ))}

              {tasks.length === 0 && !draftSelected ? (
                <div className="rounded-[12px] px-2.5 py-3 text-sm text-black/50">
                  还没有任务。点击右上角加号后开始上传图片。
                </div>
              ) : null}
            </div>
          </div>
        </aside>

        <section className="relative min-w-0 flex-1 space-y-6 pb-28 pl-4 md:pl-5">
            {isTaskLoading ? (
              <div className="absolute inset-0 z-10 flex items-center justify-center bg-[rgba(249,244,237,0.72)] backdrop-blur-[1px]">
                <div className="flex flex-col items-center gap-3 rounded-[16px] border border-[#ddcebb] bg-white/88 px-6 py-5 shadow-card">
                  <LoaderCircle className="h-8 w-8 animate-spin text-[#8a5637]" />
                  <p className="text-sm text-black/58">正在拉取任务数据…</p>
                </div>
              </div>
            ) : null}
            {isDraftView ? (
            <>
              <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.92)] px-6 py-7 shadow-card">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">New Task</p>
                <h1 className="mt-4 font-display text-5xl leading-[1.06] tracking-tight text-ink md:text-6xl">新的测试任务</h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-black/62">
                  这里先保留一个等待上传图片的空页面。任务会先创建成 draft，再分批上传图片，全部上传完成后自动进入人脸识别。
                </p>
              </section>

              <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">上传入口</p>
                    <p className="mt-2 text-base text-black/64">当前环境最多上传 {maxUploadPhotos} 张图片。图片会按 50 张一批分片上传，完成后自动启动人脸识别。</p>
                    {selectedFiles.length > 0 ? (
                      <p className="mt-3 font-mono text-xs uppercase tracking-[0.2em] text-black/42">已选择 {selectedFiles.length} / {maxUploadPhotos}</p>
                    ) : null}
                  </div>

                  <div className="flex flex-col items-start gap-3 md:items-end">
                    <label className="w-full md:min-w-[180px]">
                      <span className="mb-2 block font-mono text-[11px] uppercase tracking-[0.18em] text-black/42">
                        版本
                      </span>
                      <select
                        value={selectedTaskVersion}
                        onChange={(event) => setSelectedTaskVersion(event.target.value)}
                        disabled={isUploading}
                        className="w-full rounded-[12px] border border-[#1f1a15] bg-white/85 px-4 py-2.5 text-base text-ink outline-none disabled:cursor-not-allowed disabled:bg-black/5"
                      >
                        {availableTaskVersions.map((version) => (
                          <option key={version} value={version}>
                            {version}
                          </option>
                        ))}
                      </select>
                    </label>

                    <button
                      type="button"
                      aria-pressed={normalizeLivePhotos}
                      onClick={() => setNormalizeLivePhotos((current) => !current)}
                      disabled={isUploading}
                      className={`w-full rounded-[12px] border px-4 py-3 text-left transition md:min-w-[280px] ${
                        normalizeLivePhotos
                          ? "border-[#1f1a15] bg-white text-ink"
                          : "border-[#ddcebb] bg-white/70 text-black/58"
                      } disabled:cursor-not-allowed disabled:bg-black/5`}
                    >
                      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Live Photo 预处理</p>
                      <p className="mt-2 text-sm font-medium">
                        {normalizeLivePhotos ? "默认开启：转成静态 JPEG 再进入识别与 VLM" : "已关闭：仅在必要时转换"}
                      </p>
                      <p className="mt-1 text-xs leading-5 text-black/52">
                        原始上传仍会保留；后续如果加入 live photo 专项逻辑，可以在这个开关上继续扩展。
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isUploading}
                      className="inline-flex items-center justify-center rounded-full bg-[#1f1a15] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#2d251e] disabled:cursor-not-allowed disabled:bg-black/20"
                    >
                      选择图片并开始
                    </button>
                  </div>
                </div>

                {isUploading ? (
                  <div className="mt-5">
                    <WaitingDots label="上传完成后立即启动人脸识别" />
                  </div>
                ) : null}

                {error ? <p className="mt-4 text-sm text-[#8a5637]">{error}</p> : null}
              </section>
            </>
          ) : currentTask ? (
            <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.92)] px-6 py-7 shadow-card">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Current Task</p>
                  <h1 className="mt-4 font-display text-5xl leading-[1.06] tracking-tight text-ink md:text-6xl">{taskDisplayLabel(currentTask)}</h1>
                  <p className="mt-4 max-w-3xl text-base leading-7 text-black/62">
                    这条链路现在会把 Face to VLM to Segmentation to LLM to Memory 全部跑完。任务完成后既能看人物聚合，也能直接回看画像、关系、Event / Fact 和底层 trace。
                  </p>
                  {currentTaskSource === "api" ? (
                    <div className="mt-4 inline-flex items-center rounded-full border border-[#d7c3a8] bg-[#f6eee3] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.16em] text-[#7a5a37]">
                      API
                    </div>
                  ) : null}
                </div>

                <div className="flex flex-col items-start gap-3 lg:items-end">
                  <label className="w-full lg:min-w-[190px]">
                    <span className="sr-only">任务版本</span>
                    <select
                      value={currentTaskVersion}
                      disabled
                      className="w-full rounded-[12px] border border-[#1f1a15] bg-white/85 px-4 py-2.5 text-base text-ink outline-none disabled:cursor-not-allowed disabled:text-ink"
                    >
                      {visibleTaskVersions.map((version) => (
                        <option key={version} value={version}>
                          {version}
                        </option>
                      ))}
                    </select>
                  </label>

                  <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-5 py-4">
                    <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">状态</p>
                    <p className="mt-2 text-xl font-semibold">{currentStatusLabel}</p>
                    {showCurrentStageLabel ? <p className="mt-1 text-sm text-black/56">{currentStageLabel}</p> : null}
                  </div>

                  {currentTask.status === "uploading" ? (
                    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-5 py-4">
                      <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">上传进度</p>
                      <p className="mt-2 text-xl font-semibold">
                        {uploadTotalCount > 0 ? `已上传 ${uploadUploadedCount} / ${uploadTotalCount}` : `已上传 ${uploadUploadedCount} 张`}
                      </p>
                      <p className="mt-1 text-sm text-black/56">
                        {uploadTotalCount > 0 ? `剩余 ${uploadRemainingCount}` : "总数待确认"}
                        {uploadFailedCount > 0 ? ` · 失败 ${uploadFailedCount}` : ""}
                      </p>
                      {uploadTotalBatches > 0 ? (
                        <p className="mt-1 text-xs text-black/45">
                          分片 {Math.min(uploadCompletedBatches, uploadTotalBatches)} / {uploadTotalBatches}
                          {uploadActiveBatch > uploadCompletedBatches && uploadCompletedBatches < uploadTotalBatches
                            ? ` · 当前批次 ${uploadActiveBatch}`
                            : ""}
                          {uploadRetryAttempt > 1 ? ` · 重试 ${uploadRetryAttempt}/${UPLOAD_BATCH_RETRY_LIMIT}` : ""}
                        </p>
                      ) : null}
                      {uploadLastError ? <p className="mt-2 text-xs text-[#8a5637]">{uploadLastError}</p> : null}
                    </div>
                  ) : null}

                  {currentTask.status === "completed" && currentAnalysisBundle ? (
                    <a
                      href={toAbsoluteUrl(currentAnalysisBundle.url) ?? currentAnalysisBundle.url}
                      className="inline-flex items-center gap-2 rounded-full border border-[#1f1a15] bg-white/92 px-4 py-2.5 text-sm font-medium text-ink transition hover:bg-[#f6eee3]"
                    >
                      <Download className="h-4 w-4" />
                      <span>下载 Face / VLM / LP1</span>
                    </a>
                  ) : null}
                </div>
              </div>
            </section>
          ) : null}

          {galleryItems.length > 0 ? (
            <UploadCarousel items={galleryItems} showRecognitionBadge={showRecognitionBadge} totalCount={galleryTotalCount} />
          ) : null}

          {!isDraftView && currentTask && faceReport ? (
            <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Face Report</p>
                  <h2 className="mt-3 text-2xl font-semibold text-ink">本次人脸识别已完成</h2>
                  <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
                    共处理 {faceReport.total_images} 张图片，识别出 {faceReport.total_faces} 张脸，聚合为 {faceReport.total_persons} 个人物。
                    {faceReport.primary_person_id ? ` 主用户是 ${faceReport.primary_person_id}。` : ""}
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:min-w-[360px]">
                  <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                    <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">无脸图片</p>
                    <p className="mt-2 text-xl font-semibold text-ink">{faceReport.no_face_images.length}</p>
                  </div>
                  <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                    <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">失败图片</p>
                    <p className="mt-2 text-xl font-semibold text-ink">{faceReport.failed_images}</p>
                  </div>
                </div>
              </div>

              {faceReport.timings && (
              <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">检测耗时</p>
                  <p className="mt-2 text-xl font-semibold text-ink">{faceReport.timings.detection_seconds.toFixed(3)}s</p>
                </div>
                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">向量耗时</p>
                  <p className="mt-2 text-xl font-semibold text-ink">{faceReport.timings.embedding_seconds.toFixed(3)}s</p>
                </div>
                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">总耗时</p>
                  <p className="mt-2 text-xl font-semibold text-ink">{faceReport.timings.total_seconds.toFixed(3)}s</p>
                </div>
                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-4 py-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">平均每图</p>
                  <p className="mt-2 text-xl font-semibold text-ink">{faceReport.timings.average_image_seconds.toFixed(3)}s</p>
                </div>
              </div>
              )}

              <div className="mt-5 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">格式与 EXIF</p>
                  <div className="mt-3 space-y-2 text-sm leading-6 text-black/62">
                    <p>原始上传：原样保留，后续可直接读取完整 EXIF。</p>
                    <p>前端预览：生成 `webp` preview，仅用于浏览，不替代原图。</p>
                    <p>识别输入：{faceReport.processing.recognition_input}</p>
                    <p>识别引擎：{faceReport.engine.model_name ?? "unknown"} · {(faceReport.engine.providers ?? []).join(", ") || "unknown"}</p>
                  </div>
                </div>

                <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
                  <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">评分说明</p>
                  <div className="mt-3 space-y-2 text-sm leading-6 text-black/62">
                    <p>{faceReport.score_guide.detection_score}</p>
                    <p>{faceReport.score_guide.similarity}</p>
                  </div>
                </div>
              </div>

              <div className="mt-5 rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
                <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">精度加强</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {faceReport.precision_enhancements.map((item) => (
                    <span key={item} className="rounded-[10px] bg-[#f6eee3] px-3 py-1.5 text-sm text-[#6f5847]">
                      {item}
                    </span>
                  ))}
                </div>
              </div>

              {faceReport.no_face_images.length > 0 ? (
                <div className="mt-5">
                  <FoldableStageCard
                    title="未识别人脸图片"
                    meta={`${faceReport.no_face_images.length} 张图片本轮未识别到人脸`}
                    defaultCollapsed
                  >
                    <p className="text-sm font-medium text-[#8a5637]">以下图片本轮未识别到人脸</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {faceReport.no_face_images.map((image) => (
                        <span
                          key={image.image_id}
                          className="rounded-[10px] bg-white px-3 py-1.5 font-mono text-xs text-black/68"
                        >
                          {image.filename}
                        </span>
                      ))}
                    </div>
                  </FoldableStageCard>
                </div>
              ) : null}

              {faceReport.failed_items.length > 0 ? (
                <div className="mt-5 rounded-[12px] border border-[#e6cdbf] bg-[#fbf2ea] p-4">
                  <p className="text-sm font-medium text-[#8a5637]">以下图片在处理过程中出现失败</p>
                  <div className="mt-2 space-y-2">
                    {faceReport.failed_items.map((item) => (
                      <p key={`${item.image_id}-${item.step}`} className="text-sm leading-6 text-black/68">
                        {item.filename} [{item.step}] {item.error}
                      </p>
                    ))}
                  </div>
                </div>
              ) : null}
            </section>
          ) : null}

          {!isDraftView && currentTask ? (
            <InferencePipelinePanel
              task={currentTask}
              personIndexPanel={personIndexPanel}
              personIndexReady={personIndexReady}
            />
          ) : null}

          {!isDraftView && currentTask && currentTask.version === "v0323" ? (
            <MemoryStepsPanel
              stepsPayload={currentMemorySteps}
              loading={currentMemoryStepsLoading}
            />
          ) : null}

          {!isDraftView && currentTask && (memoryResult || currentFullMemory || currentFullMemoryLoading) ? (
            <MemoryPanel
              fullMemory={currentFullMemory}
              loading={currentFullMemoryLoading}
              queryHistory={currentQueryHistory}
            />
          ) : null}

          <RecallChatDock
            task={currentTask}
            hasMemory={Boolean(memoryResult)}
            onQueryComplete={(item) => {
              if (!currentTask) {
                return;
              }
              recordMemoryQuery(currentTask.task_id, item);
            }}
          />
        </section>
      </div>
    </main>
  );
}
