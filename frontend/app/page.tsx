"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { AlertTriangle, ArrowUp, Ban, MessageSquare, Plus, X } from "lucide-react";
import type {
  AuthResponse,
  AuthUser,
  FaceReport,
  HealthResponse,
  MemoryFocusGraph,
  MemoryQueryHistoryItem,
  MemoryQueryResponse,
  MemoryPayload,
  MilvusSegment,
  PersonGroupEntry,
  PersonGroupImage,
  TaskListResponse,
  TaskState,
  UploadItem
} from "@/lib/types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_MAX_UPLOADS = 5000;
const MAX_BATCH_FILES = 50;
const MAX_BATCH_BYTES = 64 * 1024 * 1024;
const GALLERY_PREVIEW_LIMIT = 120;
const FACE_RECOGNITION_STAGES = new Set(["queued", "starting", "loading", "converting", "face_recognition"]);
const FALLBACK_TASK_VERSIONS = ["v0317", "v0315", "v0312"];
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

function taskDisplayLabel(task: TaskState) {
  const summaryTitle =
    task.result?.summary?.title ??
    (typeof task.result_summary?.title === "string" ? task.result_summary.title : null);
  if (summaryTitle) {
    return summaryTitle;
  }
  const eventTitle = task.result?.events?.find((event) => event.title)?.title;
  if (eventTitle) {
    return eventTitle;
  }
  return task.task_id.slice(0, 8);
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
  muted = false
}: {
  label?: string;
  compact?: boolean;
  muted?: boolean;
}) {
  const tone = muted
    ? "border-black/10 bg-black/5 text-black/40"
    : "border-[#d5c4af] bg-[#f6eee3] text-[#6f5847]";

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
      <span>{label}</span>
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

function ScrollableJsonPanel({
  title,
  value,
  emptyText,
  loading = false
}: {
  title: string;
  value: unknown;
  emptyText: string;
  loading?: boolean;
}) {
  const hasValue = hasDisplayValue(value);
  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">{title}</p>
        {loading ? <WaitingDots label="处理中" compact muted /> : null}
      </div>
      {hasValue ? (
        <pre className="mt-3 max-h-[320px] overflow-auto whitespace-pre-wrap break-all rounded-[10px] bg-[#f6eee3] p-3 text-xs leading-6 text-[#5f4e42]">
          {formatJson(value)}
        </pre>
      ) : loading ? (
        <div className="mt-3">
          <WaitingDots label="等待结果生成" />
        </div>
      ) : (
        <p className="mt-3 text-sm text-black/56">{emptyText}</p>
      )}
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

function InferencePipelinePanel({ task }: { task: TaskState }) {
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
  const memoryStage = readStageProgress(task.progress, "memory");

  const faceValue = task.result?.face_recognition ?? faceStage.face_result_preview ?? null;
  const vlmValue =
    (vlmStage.vlm_results_preview as unknown) ??
    task.result?.memory?.transparency?.vlm_stage?.summaries ??
    null;
  const llmValue =
    task.result?.memory_contract ??
    (llmStage.memory_contract_preview as unknown) ??
    null;
  const redisValue =
    task.result?.memory?.storage?.redis ??
    (memoryStage.redis_preview as unknown) ??
    null;
  const neo4jValue =
    task.result?.memory?.storage?.neo4j ??
    (memoryStage.neo4j_preview as unknown) ??
    null;

  const vlmLoading = task.status === "running" && currentStage === "vlm";
  const llmLoading = task.status === "running" && currentStage === "llm";
  const memoryLoading = task.status === "running" && currentStage === "memory";

  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Inference Timeline</p>
          <h2 className="mt-3 text-2xl font-semibold text-ink">推理生成逐阶段展开</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
            前端会按 `Face → VLM → LLM → Redis → Neo4j` 顺序展示当前任务的阶段结果，并在运行阶段保留 3 dots 提示。
          </p>
        </div>
        <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-5 py-4">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">Current Stage</p>
          <p className="mt-2 text-xl font-semibold">{formatStage(currentStage)}</p>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <ScrollableJsonPanel
          title="Face Results"
          value={faceValue}
          emptyText="当前还没有可展示的人脸识别结果。"
          loading={task.status === "running" && FACE_RECOGNITION_STAGES.has(currentStage) && !hasDisplayValue(faceValue)}
        />
        <ScrollableJsonPanel
          title="VLM Results"
          value={vlmValue}
          emptyText="VLM 结果尚未产出。"
          loading={vlmLoading}
        />
        <ScrollableJsonPanel
          title="LLM Results"
          value={llmValue}
          emptyText="LLM 结果尚未产出。"
          loading={llmLoading}
        />
        <ScrollableJsonPanel
          title="Redis Profile Materialization"
          value={redisValue}
          emptyText="Redis 画像数据尚未产出。"
          loading={memoryLoading && !hasDisplayValue(redisValue)}
        />
        <ScrollableJsonPanel
          title="Neo4j Graph Materialization"
          value={neo4jValue}
          emptyText="Neo4j graph 数据尚未产出。"
          loading={memoryLoading && !hasDisplayValue(neo4jValue)}
        />
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

function GraphLandingPanel({
  memory,
  taskVersion,
}: {
  memory: MemoryPayload;
  taskVersion?: string | null;
}) {
  const nodes = memory.storage?.neo4j?.nodes ?? {};
  const edges = memory.storage?.neo4j?.edges ?? [];
  const eventNodes = sortByRecent((nodes.events ?? []) as Array<Record<string, unknown>>, (item) =>
    String(item.started_at ?? item.ended_at ?? "")
  );
  const relationshipNodes = sortByRecent(
    (nodes.relationship_hypotheses ?? []) as Array<Record<string, unknown>>,
    (item) => String(item.window_end ?? item.window_start ?? "")
  );

  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Neo4j Landing</p>
          <p className="mt-2 text-sm leading-6 text-black/60">
            当前以 `v0317` 的 canonical graph 方式落位，Photo 不进图，图内只保留 facts + hypotheses。
          </p>
        </div>
        <div className="rounded-[10px] bg-[#f6eee3] px-3 py-2 text-xs leading-6 text-[#5f4e42]">
          {taskVersion ?? "unknown"} · {Object.keys(nodes).length} groups · {edges.length} edges
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {Object.entries(memory.transparency.neo4j_state?.node_counts ?? {}).map(([label, count]) => (
          <MetricCard key={label} label={label} value={count} />
        ))}
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <StageSummaryPanel
          title="Event Nodes"
          items={eventNodes}
          emptyText="当前还没有 Event hypothesis 落位。"
        />
        <StageSummaryPanel
          title="Relationship Nodes"
          items={relationshipNodes}
          emptyText="当前还没有 RelationshipHypothesis 落位。"
        />
      </div>
    </div>
  );
}

function MilvusLandingPanel({ segments }: { segments: MilvusSegment[] }) {
  const orderedSegments = sortByRecent(segments, (segment) => segment.started_at ?? segment.ended_at ?? "");

  return (
    <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Milvus Landing</p>
      <p className="mt-2 text-sm leading-6 text-black/60">
        Milvus 负责 evidence segments，当前已携带 `started_at / ended_at / place_uuid / location_hint` 这类检索辅助 metadata。
      </p>
      <div className="mt-4 space-y-3">
        {orderedSegments.length > 0 ? (
          orderedSegments.slice(0, 8).map((segment) => (
            <div key={segment.segment_uuid} className="rounded-[10px] bg-[#f6eee3] px-3 py-3">
              <div className="flex flex-wrap items-center gap-2 text-xs text-[#6f5847]">
                <span className="rounded-full bg-white px-2 py-1 font-mono">{segment.segment_type}</span>
                {segment.started_at ? <span>{formatDateTime(segment.started_at)}</span> : null}
                {segment.location_hint ? <span>{segment.location_hint}</span> : null}
              </div>
              <p className="mt-2 text-sm leading-6 text-[#5f4e42]">{segment.text}</p>
              <p className="mt-2 text-xs leading-6 text-[#7a6858]">
                session {segment.session_uuid ?? "n/a"} · event {segment.event_uuid ?? "n/a"} · relationship{" "}
                {segment.relationship_uuid ?? "n/a"}
              </p>
            </div>
          ))
        ) : (
          <p className="text-sm text-black/56">当前还没有落到 Milvus 的 evidence segments。</p>
        )}
      </div>
    </div>
  );
}

function truncateGraphLabel(value: string, max = 18) {
  if (value.length <= max) {
    return value;
  }
  return `${value.slice(0, max - 1)}…`;
}

function graphNodeColor(nodeType: string) {
  switch (nodeType) {
    case "primary_person":
      return { fill: "#d85a31", stroke: "#8d2c0d" };
    case "related_person":
    case "participant_person":
      return { fill: "#f3b470", stroke: "#9d6425" };
    case "event":
      return { fill: "#7bb88f", stroke: "#356f49" };
    case "session":
      return { fill: "#8eb7c8", stroke: "#426e80" };
    case "timeline":
      return { fill: "#c8a6d6", stroke: "#6f4c7d" };
    case "photo":
      return { fill: "#e7d7bc", stroke: "#8c7357" };
    default:
      return { fill: "#d8c9b7", stroke: "#786657" };
  }
}

function FocusGraphPanel({ graph }: { graph?: MemoryFocusGraph | null }) {
  const layout = useMemo(() => {
    if (!graph || graph.nodes.length === 0) {
      return null;
    }

    const width = 820;
    const height = 560;
    const centerX = width / 2;
    const centerY = height / 2;
    const ringRadii: Record<number, number> = { 0: 0, 1: 150, 2: 245, 3: 330 };
    const grouped = new Map<number, MemoryFocusGraph["nodes"]>();
    for (const node of graph.nodes) {
      const ring = Math.max(0, Math.min(3, Number(node.ring ?? 0)));
      const bucket = grouped.get(ring) ?? [];
      bucket.push(node);
      grouped.set(ring, bucket);
    }

    const positions = new Map<string, { x: number; y: number }>();
    for (const [ring, nodes] of grouped.entries()) {
      const ordered = [...nodes].sort((left, right) => {
        if (left.node_id === graph.center_node_id) {
          return -1;
        }
        if (right.node_id === graph.center_node_id) {
          return 1;
        }
        if (left.node_type === "user_account") {
          return -1;
        }
        if (right.node_type === "user_account") {
          return 1;
        }
        return left.label.localeCompare(right.label);
      });

      if (ring === 0 || ordered.length === 1) {
        positions.set(ordered[0].node_id, { x: centerX, y: centerY });
        continue;
      }

      const radius = ringRadii[ring] ?? 150;
      const angleStep = (Math.PI * 2) / ordered.length;
      const startAngle = -Math.PI / 2;
      ordered.forEach((node, index) => {
        const angle = startAngle + angleStep * index;
        positions.set(node.node_id, {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius
        });
      });
    }

    return {
      width,
      height,
      centerX,
      centerY,
      ringRadii,
      positions
    };
  }, [graph]);

  if (!graph || !layout) {
    return null;
  }

  return (
    <div className="mt-5 rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Primary-Centered Graph</p>
          <p className="mt-2 text-sm leading-6 text-black/60">
            这张图以 `primary user` 为中心发散，优先展示主角、关系对象、事件、会话和时间线。
          </p>
        </div>
        <div className="rounded-[10px] bg-[#f6eee3] px-3 py-2 text-xs leading-6 text-[#5f4e42]">
          nodes {graph.node_count} · edges {graph.edge_count} · primary {graph.primary_face_person_id ?? "unresolved"}
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <svg
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="h-[560px] w-full min-w-[760px] rounded-[12px] bg-[#f8f1e8]"
        >
          {[1, 2, 3].map((ring) => (
            <circle
              key={ring}
              cx={layout.centerX}
              cy={layout.centerY}
              r={layout.ringRadii[ring]}
              fill="none"
              stroke="#e3d5c5"
              strokeDasharray="8 10"
            />
          ))}

          {graph.edges.map((edge) => {
            const source = layout.positions.get(edge.source_id);
            const target = layout.positions.get(edge.target_id);
            if (!source || !target) {
              return null;
            }
            const labelX = (source.x + target.x) / 2;
            const labelY = (source.y + target.y) / 2;
            return (
              <g key={edge.edge_id}>
                <line
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="#b89976"
                  strokeWidth={edge.edge_type === "RELATIONSHIP_HYPOTHESIS" ? 2.4 : 1.8}
                  opacity={0.9}
                />
                <text
                  x={labelX}
                  y={labelY}
                  textAnchor="middle"
                  className="fill-[#7b6654] text-[10px] font-mono uppercase tracking-[0.12em]"
                >
                  {truncateGraphLabel(edge.label || edge.edge_type, 14)}
                </text>
              </g>
            );
          })}

          {graph.nodes.map((node) => {
            const position = layout.positions.get(node.node_id);
            if (!position) {
              return null;
            }
            const palette = graphNodeColor(node.node_type);
            const radius = node.is_primary ? 34 : node.node_type === "event" ? 26 : 22;
            return (
              <g key={node.node_id}>
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={radius}
                  fill={palette.fill}
                  stroke={palette.stroke}
                  strokeWidth={node.is_primary ? 4 : 2}
                />
                <text
                  x={position.x}
                  y={position.y - 2}
                  textAnchor="middle"
                  className="fill-[#1d130c] text-[11px] font-semibold"
                >
                  {truncateGraphLabel(node.label, 16)}
                </text>
                <text
                  x={position.x}
                  y={position.y + 14}
                  textAnchor="middle"
                  className="fill-[#3d2f25] text-[9px] font-mono uppercase tracking-[0.12em]"
                >
                  {node.node_type}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-4 grid gap-2 md:grid-cols-3 xl:grid-cols-6">
        {[
          ["primary_person", "Primary"],
          ["related_person", "Related"],
          ["event", "Event"],
          ["session", "Session"],
          ["timeline", "Timeline"],
          ["photo", "Photo"]
        ].map(([nodeType, label]) => {
          const palette = graphNodeColor(nodeType);
          return (
            <div key={nodeType} className="flex items-center gap-2 rounded-[10px] bg-[#f6eee3] px-3 py-2 text-xs text-[#5f4e42]">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: palette.fill, border: `1px solid ${palette.stroke}` }}
              />
              {label}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MemoryPanel({
  memory,
  profileMarkdown,
  taskVersion,
  queryHistory
}: {
  memory: MemoryPayload;
  profileMarkdown?: string | null;
  taskVersion?: string | null;
  queryHistory: MemoryQueryHistoryItem[];
}) {
  const profileFields = memory.storage?.redis?.profile_core?.fields ?? {};
  const relationships = memory.storage?.redis?.profile_relationships?.items ?? [];
  const recentEvents = sortByRecent(
    (memory.storage?.redis?.profile_recent_events?.items ?? []) as Array<Record<string, unknown>>,
    (item) => String(item.started_at ?? item.generated_at ?? "")
  );
  const externalPublish = memory.external_publish ?? null;
  const focusGraph = memory.transparency.focus_graph ?? memory.storage?.neo4j?.focus_graph ?? null;
  const milvusSegments = memory.storage?.milvus?.segments ?? [];
  const stageVlmSummaries = (memory.transparency.vlm_stage?.summaries ?? []) as Array<Record<string, unknown>>;
  const stageSequenceSummaries = (memory.transparency.sequence_stage?.summaries ?? []) as Array<Record<string, unknown>>;
  const stageLlmSummaries = (memory.transparency.llm_stage?.summaries ?? []) as Array<Record<string, unknown>>;
  const recentQueries = sortByRecent(queryHistory, (item) => item.requested_at);
  const segmentById = new Map(milvusSegments.map((segment) => [segment.segment_uuid, segment]));
  const stageCards = [
    {
      key: "face",
      label: "Face",
      value: memory.transparency.face_stage?.total_faces ?? 0,
      detail: `${memory.transparency.face_stage?.total_persons ?? 0} persons`,
    },
    {
      key: "vlm",
      label: "VLM",
      value: memory.transparency.vlm_stage?.processed_photos ?? 0,
      detail: `${memory.transparency.vlm_stage?.cached_hits ?? 0} cached`,
    },
    {
      key: "sequence",
      label: "Sequence",
      value: memory.transparency.sequence_stage?.session_count ?? 0,
      detail: `${memory.transparency.sequence_stage?.timeline_count ?? 0} timelines`,
    },
    {
      key: "llm",
      label: "LLM",
      value: memory.transparency.llm_stage?.event_candidate_count ?? 0,
      detail: `${memory.transparency.llm_stage?.relationship_hypothesis_count ?? 0} relationships`,
    },
    {
      key: "neo4j",
      label: "Neo4j",
      value: memory.transparency.neo4j_state?.edge_count ?? 0,
      detail: `${Object.values(memory.transparency.neo4j_state?.node_counts ?? {}).reduce((sum, count) => sum + count, 0)} nodes`,
    },
    {
      key: "milvus",
      label: "Milvus",
      value: memory.transparency.milvus_state?.segment_count ?? 0,
      detail: `${Object.keys(memory.transparency.milvus_state?.segment_type_counts ?? {}).length} segment types`,
    },
    {
      key: "redis",
      label: "Redis",
      value: memory.transparency.redis_state?.published_field_count ?? 0,
      detail: `profile v${memory.transparency.redis_state?.profile_version ?? 0}`,
    },
    {
      key: "changes",
      label: "Changes",
      value: memory.transparency.object_diff?.change_count ?? 0,
      detail: `${(memory.transparency.publish_decisions ?? []).length} publish decisions`,
    },
  ];

  return (
    <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Memory Transparency</p>
          <h2 className="mt-3 text-2xl font-semibold text-ink">记忆框架已物化到任务结果</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-black/60">
            这里直接展开 `Face / VLM / Sequence / LLM / Neo4j / Milvus / Redis / changes` 八层结果，方便做画像、关系和事件精度回看。
          </p>
        </div>
        <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 px-5 py-4">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">Profile Version</p>
          <p className="mt-2 text-xl font-semibold">{memory.summary.profile_version}</p>
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Photos" value={memory.summary.photo_count} detail={`${memory.summary.person_count} persons`} />
        <MetricCard label="Sessions" value={memory.summary.session_count} detail={`${memory.summary.timeline_count} timelines`} />
        <MetricCard label="Events" value={memory.summary.event_candidate_count} detail={`${memory.summary.relationship_count} relationships`} />
        <MetricCard label="Profile Fields" value={memory.summary.profile_field_count} detail={`${memory.summary.segment_count} segments`} />
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-3">
        <StageSummaryPanel
          title="VLM Outputs"
          items={stageVlmSummaries}
          emptyText="当前还没有可展示的 VLM stage summaries。"
        />
        <StageSummaryPanel
          title="Sequence Outputs"
          items={stageSequenceSummaries}
          emptyText="当前还没有 sequence stage summaries。"
        />
        <StageSummaryPanel
          title="LLM Outputs"
          items={stageLlmSummaries}
          emptyText="当前还没有可展示的 LLM stage summaries。"
        />
      </div>

      {externalPublish ? (
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <MetricCard
            label="Redis Publish"
            value={externalPublish.redis?.status ?? "unknown"}
            detail={externalPublish.redis?.reason ?? `${externalPublish.redis?.key_count ?? 0} keys`}
          />
          <MetricCard
            label="Neo4j Publish"
            value={externalPublish.neo4j?.status ?? "unknown"}
            detail={externalPublish.neo4j?.reason ?? `${externalPublish.neo4j?.edge_count ?? 0} edges`}
          />
          <MetricCard
            label="Milvus Publish"
            value={externalPublish.milvus?.status ?? "unknown"}
            detail={externalPublish.milvus?.reason ?? `${externalPublish.milvus?.segment_count ?? 0} segments`}
          />
        </div>
      ) : null}

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {stageCards.map((item) => (
          <MetricCard key={item.key} label={item.label} value={item.value} detail={item.detail} />
        ))}
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Redis Profile</p>
          <div className="mt-3 space-y-3">
            {Object.entries(profileFields).length > 0 ? (
              Object.entries(profileFields).map(([fieldKey, field]) => (
                <div key={fieldKey} className="rounded-[10px] bg-[#f6eee3] px-3 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-[#6f5847]">{fieldKey}</p>
                    <p className="text-xs text-[#6f5847]">confidence {field.confidence.toFixed(2)}</p>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-[#5f4e42]">{field.values.join(" / ") || "No values"}</p>
                  <p className="mt-2 text-xs text-[#7a6858]">
                    events: {field.supporting_event_ids.length} · refs: {field.evidence_refs.length}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-black/56">当前还没有可发布到 Redis 的画像字段。</p>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Relationships</p>
            <div className="mt-3 space-y-2">
              {relationships.length > 0 ? (
                relationships.slice(0, 6).map((item, index) => (
                  <div key={`${String(item.person_uuid ?? item.face_person_id)}-${index}`} className="rounded-[10px] bg-[#f6eee3] px-3 py-3 text-sm leading-6 text-[#5f4e42]">
                    {String(item.face_person_id ?? "unknown")} · {String(item.label ?? "unknown")} · {Number(item.confidence ?? 0).toFixed(2)}
                  </div>
                ))
              ) : (
                <p className="text-sm text-black/56">当前还没有人物关系候选。</p>
              )}
            </div>
          </div>

          <div className="rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Recent Events</p>
            <div className="mt-3 space-y-2">
              {recentEvents.length > 0 ? (
                recentEvents.slice(0, 6).map((item, index) => (
                  <div key={`${String(item.event_uuid ?? item.event_id)}-${index}`} className="rounded-[10px] bg-[#f6eee3] px-3 py-3 text-sm leading-6 text-[#5f4e42]">
                    {String(item.title ?? "Untitled")} · {String(item.location ?? "unknown")}
                  </div>
                ))
              ) : (
                <p className="text-sm text-black/56">当前还没有事件候选。</p>
              )}
            </div>
          </div>
        </div>
      </div>

      <FocusGraphPanel graph={focusGraph} />

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <GraphLandingPanel memory={memory} taskVersion={taskVersion} />
        <MilvusLandingPanel segments={milvusSegments} />
      </div>

      <div className="mt-5 rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Memory Debug / Recall Chain</p>
            <p className="mt-2 text-sm leading-6 text-black/60">
              底部 chat bar 发起的问题会在这里按生成时间倒序展开，展示 `OperatorPlan → Recall → Graph → Milvus → Answer` 整条链路。
            </p>
          </div>
          <div className="rounded-[10px] bg-[#f6eee3] px-3 py-2 text-xs leading-6 text-[#5f4e42]">
            recent first · {recentQueries.length} queries
          </div>
        </div>

        <div className="mt-4 space-y-4">
          {recentQueries.length > 0 ? (
            recentQueries.map((item) => {
              const answer = item.response.answer;
              const trace = item.response.debug_trace;
              const evidenceSegments = answer.evidence_segment_ids
                .map((segmentId) => segmentById.get(segmentId))
                .filter((segment): segment is MilvusSegment => Boolean(segment));
              return (
                <article key={item.query_id} className="rounded-[12px] border border-[#eadbcc] bg-[#fcf8f3] p-4">
                  <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                    <div>
                      <p className="text-base font-semibold text-ink">{item.question}</p>
                      <p className="mt-1 text-xs text-black/50">{formatDateTime(item.requested_at)}</p>
                    </div>
                    <div className="rounded-[10px] bg-white px-3 py-2 text-xs leading-6 text-[#5f4e42]">
                      confidence {answer.confidence.toFixed(2)}
                    </div>
                  </div>

                  <div className="mt-3 rounded-[10px] bg-white px-4 py-3">
                    <p className="text-sm leading-6 text-[#5f4e42]">{answer.summary || "暂无摘要输出"}</p>
                    {answer.uncertainty_flags.length > 0 ? (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {answer.uncertainty_flags.map((flag) => (
                          <span key={flag} className="rounded-full bg-[#f6eee3] px-2.5 py-1 text-xs text-[#7a6858]">
                            {flag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>

                  <div className="mt-3 grid gap-3 xl:grid-cols-2">
                    <div className="rounded-[10px] bg-white px-4 py-3">
                      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-black/42">Graph Hits</p>
                      <p className="mt-2 text-xs leading-6 text-[#7a6858]">
                        events {answer.supporting_events.length} · sessions {answer.supporting_sessions.length} · relationships {answer.supporting_relationships.length}
                      </p>
                      <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-all text-xs leading-6 text-[#5f4e42]">
                        {formatJson({
                          resolved_concepts: answer.resolved_concepts,
                          resolved_entities: answer.resolved_entities,
                          supporting_events: answer.supporting_events,
                          supporting_sessions: answer.supporting_sessions,
                          supporting_relationships: answer.supporting_relationships
                        })}
                      </pre>
                    </div>

                    <div className="rounded-[10px] bg-white px-4 py-3">
                      <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-black/42">Milvus Evidence</p>
                      {evidenceSegments.length > 0 ? (
                        <div className="mt-2 space-y-2">
                          {evidenceSegments.slice(0, 6).map((segment) => (
                            <div key={segment.segment_uuid} className="rounded-[10px] bg-[#f6eee3] px-3 py-3">
                              <p className="text-xs text-[#7a6858]">
                                {segment.segment_type} · {segment.location_hint || "unknown place"} · {formatDateTime(segment.started_at ?? segment.ended_at ?? undefined)}
                              </p>
                              <p className="mt-2 text-sm leading-6 text-[#5f4e42]">{segment.text}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="mt-2 text-sm text-black/56">当前回答没有额外命中 Milvus evidence。</p>
                      )}
                    </div>
                  </div>

                  <div className="mt-3 grid gap-3 xl:grid-cols-2">
                    <JsonDetails title="Operator Plan" value={trace.operator_plan} />
                    <JsonDetails title="Recall Candidates" value={trace.recall_candidates} />
                    <JsonDetails title="Query DSL" value={trace.dsl} />
                    <JsonDetails title="Pseudo Cypher / Evidence Fill" value={{ executed_cypher: trace.executed_cypher, evidence_fill: trace.evidence_fill }} />
                  </div>
                </article>
              );
            })
          ) : (
            <p className="text-sm text-black/56">记忆落位后，可以直接在底部 chat bar 里提问；链路会显示在这里。</p>
          )}
        </div>
      </div>

      {profileMarkdown ? (
        <div className="mt-5 rounded-[12px] border border-[#ddcebb] bg-white/70 p-4">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-black/42">Profile Markdown</p>
          <pre className="mt-3 max-h-[360px] overflow-auto whitespace-pre-wrap rounded-[10px] bg-[#f6eee3] p-3 text-sm leading-6 text-[#5f4e42]">
            {profileMarkdown}
          </pre>
        </div>
      ) : null}

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <JsonDetails title="Envelope DTO" value={memory.envelope} />
        <JsonDetails title="Neo4j / Milvus / Redis State" value={memory.storage} />
        <JsonDetails title="Transparency Views" value={memory.transparency} />
        {focusGraph ? <JsonDetails title="Focus Graph Mermaid" value={focusGraph.mermaid} /> : null}
        <JsonDetails title="Evaluation Objects" value={memory.evaluation} />
        {externalPublish ? <JsonDetails title="External Publish Report" value={externalPublish} /> : null}
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
  const [availableTaskVersions, setAvailableTaskVersions] = useState<string[]>(FALLBACK_TASK_VERSIONS);
  const [defaultTaskVersion, setDefaultTaskVersion] = useState(FALLBACK_DEFAULT_TASK_VERSION);
  const [selectedTaskVersion, setSelectedTaskVersion] = useState(FALLBACK_DEFAULT_TASK_VERSION);
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
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [memoryQueryHistoryByTask, setMemoryQueryHistoryByTask] = useState<Record<string, MemoryQueryHistoryItem[]>>({});
  const [error, setError] = useState<string | null>(null);

  const personGroups = currentTask?.result?.face_recognition?.person_groups ?? [];
  const faceReport = useMemo(() => normalizeFaceReport(currentTask?.result?.face_report ?? null), [currentTask]);
  const memoryResult = currentTask?.result?.memory ?? null;
  const currentUploads = currentTask?.uploads ?? [];
  const taskGroups = useMemo(() => groupTasksByCreatedAt(tasks), [tasks]);
  const currentQueryHistory = currentTask ? memoryQueryHistoryByTask[currentTask.task_id] ?? [] : [];
  const currentTaskVersion = currentTask?.version ?? LEGACY_TASK_VERSION;
  const visibleTaskVersions = useMemo(() => Array.from(new Set([currentTaskVersion, ...availableTaskVersions])), [availableTaskVersions, currentTaskVersion]);
  const currentStatusLabel = currentTask ? formatStatus(currentTask.status) : "";
  const currentStageLabel = currentTask ? formatStage(currentTask.stage) : "";
  const showCurrentStageLabel = Boolean(currentTask && currentStageLabel !== currentStatusLabel);

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
    const orderedTasks = sortTasksByCreatedAt(payload.tasks);
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

  async function fetchTask(taskId: string) {
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
    setTasks((previous) => {
      const taskIndex = previous.findIndex((task) => task.task_id === payload.task_id);
      if (taskIndex === -1) {
        return sortTasksByCreatedAt([...previous, payload]);
      }

      const next = [...previous];
      next[taskIndex] = payload;
      return next;
    });
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
    if (isDraftView || !currentTask || (currentTask.status !== "queued" && currentTask.status !== "running")) {
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

    try {
      const response = await apiFetch(`${API_BASE}/api/tasks`, {
        method: "POST",
        body: JSON.stringify({
          version: selectedTaskVersion
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
      setIsDraftView(false);
      await fetchTask(payload.task_id);

      const batches = buildUploadBatches(files);
      let cursor = 0;
      const uploadWorker = async () => {
        while (cursor < batches.length) {
          const batch = batches[cursor];
          cursor += 1;
          const formData = new FormData();
          batch.forEach((file) => formData.append("files", file));
          const batchResponse = await apiFetch(`${API_BASE}/api/tasks/${payload.task_id}/upload-batches`, {
            method: "POST",
            body: formData
          });
          if (!batchResponse.ok) {
            const batchPayload = await batchResponse.json().catch(() => null);
            throw new Error(batchPayload?.detail ?? "上传图片分片失败");
          }
        }
      };
      await Promise.all(Array.from({ length: Math.min(2, batches.length) }, () => uploadWorker()));

      const startResponse = await apiFetch(`${API_BASE}/api/tasks/${payload.task_id}/start`, {
        method: "POST",
        body: JSON.stringify({
          max_photos: Math.min(files.length, maxUploadPhotos),
          use_cache: false
        })
      });
      if (!startResponse.ok) {
        const startPayload = await startResponse.json().catch(() => null);
        throw new Error(startPayload?.detail ?? "启动任务失败");
      }

      await fetchTasks();
      await fetchTask(payload.task_id);
      setPendingUploads([]);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "创建任务失败");
      if (createdTaskId) {
        await fetchTask(createdTaskId).catch(() => null);
      }
    } finally {
      setIsUploading(false);
      setSelectedFiles([]);
      if (createdTaskId && fileInputRef.current) {
        fileInputRef.current.value = "";
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

  return (
    <main className="min-h-screen px-2 py-6 md:px-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
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
                            onClick={() => fetchTask(task.task_id).catch(() => null)}
                            className="min-w-0 flex-1 text-left"
                          >
                            <p className="truncate text-sm font-medium text-ink">{taskDisplayLabel(task)}</p>
                            <p className="mt-1 truncate text-xs text-black/45">
                              {formatStatus(task.status)} · {formatTaskTime(task.created_at) || formatStage(task.stage)}
                            </p>
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

        <section className="min-w-0 flex-1 space-y-6 pb-28 pl-4 md:pl-5">
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
                    这条链路现在会把 Face to VLM to Sequence to LLM to Memory 全部跑完。任务完成后既能看人物聚合，也能直接回看画像、关系、事件和底层 trace。
                  </p>
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
                <div className="mt-5 rounded-[12px] border border-[#e6cdbf] bg-[#fbf2ea] p-4">
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
            <InferencePipelinePanel task={currentTask} />
          ) : null}

          {!isDraftView && currentTask && memoryResult ? (
            <MemoryPanel
              memory={memoryResult}
              profileMarkdown={currentTask.result?.profile_markdown}
              taskVersion={currentTask.version}
              queryHistory={currentQueryHistory}
            />
          ) : null}

          {!isDraftView && currentTask ? (
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
            ) : currentTask.status === "running" || currentTask.status === "queued" || currentTask.status === "uploading" ? (
              <WaitingDots label={`当前阶段：${formatStage(currentTask.stage)}`} />
            ) : (
              <section className="w-full rounded-[12px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-8 shadow-card">
                <p className="text-lg font-medium">还没有可展示的人脸识别结果</p>
                <p className="mt-2 text-sm text-black/58">
                  当前任务尚未产出可用的人物聚合结果。
                </p>
              </section>
            )
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
