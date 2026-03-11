"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { ArrowUp, Plus } from "lucide-react";
import type { FailureItem, TaskListResponse, TaskState, UploadItem } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_UPLOADS = 100;
const FACE_RECOGNITION_STAGES = new Set(["queued", "starting", "loading", "converting", "face_recognition"]);

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

const statusLabelMap: Record<TaskState["status"], string> = {
  queued: "排队中",
  running: "处理中",
  completed: "已完成",
  failed: "失败"
};

const stageLabelMap: Record<string, string> = {
  queued: "等待执行",
  starting: "初始化",
  loading: "读取图片",
  converting: "格式转换",
  face_recognition: "人脸识别",
  preprocess: "图片压缩",
  vlm: "视觉分析",
  llm: "推理生成",
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

function taskDisplayLabel(task: TaskState) {
  const summaryTitle = task.result?.summary?.title;
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
  return files.map((file, index) => ({
    id: `pending-${index}-${file.name}-${file.size}-${file.lastModified}`,
    filename: file.name,
    previewUrl: URL.createObjectURL(file),
    sizeLabel: formatBytes(file.size)
  }));
}

function revokePendingUploads(items: PendingUpload[]) {
  items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
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

function UploadCarousel({
  items,
  showRecognitionBadge
}: {
  items: GalleryCard[];
  showRecognitionBadge: boolean;
}) {
  return (
    <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.9)] p-6 shadow-card">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">任务图片走廊</p>
          <p className="mt-2 text-sm leading-6 text-black/60">这里保留刚上传或已入库的图片缩略图、名称和格式信息，方便测试同学快速核对任务内容。</p>
        </div>
        <div className="text-sm text-black/45">横向滚动查看全部图片</div>
      </div>

      <div className="mt-5 flex snap-x gap-4 overflow-x-auto pb-2">
        {items.map((item) => (
          <article
            key={item.id}
            className="relative min-w-[260px] snap-start overflow-hidden rounded-[14px] border border-[#ddcebb] bg-[#f7f0e6]"
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

function FailureList({ failures }: { failures: FailureItem[] }) {
  return (
    <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">坏图与失败记录</p>
        <span className="rounded-[10px] bg-[#ead8ca] px-3 py-1 font-mono text-xs text-[#8a5637]">{failures.length}</span>
      </div>

      {failures.length === 0 ? (
        <p className="mt-3 text-sm text-black/65">当前任务没有记录到坏图或处理失败图片。</p>
      ) : (
        <div className="mt-4 space-y-3">
          {failures.map((failure) => (
            <div
              key={`${failure.image_id}-${failure.step}-${failure.filename}`}
              className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{failure.filename}</span>
                <span className="rounded-[10px] bg-black/5 px-2 py-1 font-mono text-xs">{failure.image_id}</span>
                <span className="rounded-[10px] bg-black/5 px-2 py-1 font-mono text-xs">{failure.step}</span>
              </div>
              <p className="mt-2 text-sm leading-6 text-black/65">{failure.error}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function RecallChatDock() {
  return (
    <div className="fixed bottom-4 left-4 right-4 z-40 md:left-[348px] md:right-6">
      <div className="mx-auto max-w-[1320px] rounded-[18px] border border-[#d8c9b7] bg-[rgba(248,243,236,0.96)] p-4 shadow-card backdrop-blur">
        <div className="flex items-end gap-3">
          <textarea
            disabled
            rows={1}
            placeholder="记忆布局完成后可在这里输入召回问题"
            className="h-11 max-h-[72px] min-h-[44px] flex-1 resize-none overflow-y-auto rounded-[14px] border border-black/8 bg-[#f4efe7] px-4 py-[10px] text-sm leading-6 text-black/40 outline-none"
          />
          <button
            type="button"
            disabled
            aria-label="发送"
            className="inline-flex h-11 w-11 items-center justify-center rounded-[14px] border border-black/10 bg-[#ebe4d8] text-black/35"
          >
            <ArrowUp size={16} strokeWidth={2} />
          </button>
        </div>

        <p className="mt-3 px-1 text-xs text-black/45">记忆布局完成后可以进行召回测试</p>
      </div>
    </div>
  );
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [tasks, setTasks] = useState<TaskState[]>([]);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
  const [isDraftView, setIsDraftView] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const faceRecognition = currentTask?.result?.face_recognition;
  const imageEntries = faceRecognition?.images ?? [];
  const failures = currentTask?.result?.failed_images ?? [];
  const warnings = currentTask?.result?.warnings ?? [];
  const currentUploads = currentTask?.uploads ?? [];

  useEffect(() => {
    return () => {
      revokePendingUploads(pendingUploads);
    };
  }, [pendingUploads]);

  const stats = useMemo(() => {
    if (!currentTask?.result) {
      return [];
    }

    return [
      { label: "已提交图片", value: currentTask.result.summary.total_uploaded },
      { label: "识别人脸", value: currentTask.result.summary.total_faces },
      { label: "识别人物", value: currentTask.result.summary.total_persons },
      { label: "失败图片", value: currentTask.result.summary.failed_images }
    ];
  }, [currentTask]);

  const galleryItems = useMemo<GalleryCard[]>(() => {
    if (pendingUploads.length > 0) {
      return pendingUploads.map((item) => ({
        id: item.id,
        filename: item.filename,
        imageUrl: item.previewUrl,
        meta: item.sizeLabel
      }));
    }

    return currentUploads.map((upload) => ({
      id: upload.stored_filename,
      filename: upload.filename,
      imageUrl: toAbsoluteUrl(upload.url),
      meta: uploadMeta(upload)
    }));
  }, [currentUploads, pendingUploads]);

  const showRecognitionBadge =
    pendingUploads.length > 0 ||
    Boolean(
      currentTask &&
        currentTask.status !== "completed" &&
        currentTask.status !== "failed" &&
        FACE_RECOGNITION_STAGES.has(currentTask.stage)
    );

  async function fetchTasks(options?: { selectInitial?: boolean; preserveCurrent?: boolean }) {
    const selectInitial = options?.selectInitial ?? false;
    const preserveCurrent = options?.preserveCurrent ?? true;

    const response = await fetch(`${API_BASE}/api/tasks?limit=30`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("获取任务列表失败");
    }
    const payload = (await response.json()) as TaskListResponse;
    setTasks(payload.tasks);

    if (selectInitial) {
      if (payload.tasks.length > 0) {
        setCurrentTask(payload.tasks[0]);
        setIsDraftView(false);
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
      const matched = payload.tasks.find((task) => task.task_id === currentTask.task_id);
      if (matched) {
        setCurrentTask(matched);
      }
    }
  }

  async function fetchTask(taskId: string) {
    const response = await fetch(`${API_BASE}/api/tasks/${taskId}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("读取任务详情失败");
    }
    const payload = (await response.json()) as TaskState;
    setCurrentTask(payload);
    setIsDraftView(false);
    setTasks((previous) => {
      const next = previous.filter((task) => task.task_id !== payload.task_id);
      return [payload, ...next];
    });
  }

  useEffect(() => {
    fetchTasks({ selectInitial: true }).catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : "初始化任务列表失败");
    });
  }, []);

  useEffect(() => {
    if (isDraftView || !currentTask || currentTask.status === "completed" || currentTask.status === "failed") {
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
    let created = false;

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      formData.append("max_photos", String(Math.min(files.length, MAX_UPLOADS)));
      formData.append("use_cache", "false");

      const response = await fetch(`${API_BASE}/api/tasks`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "创建任务失败");
      }

      const payload = (await response.json()) as { task_id: string };
      await fetchTask(payload.task_id);
      await fetchTasks();
      setPendingUploads([]);
      created = true;
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "创建任务失败");
    } finally {
      setIsUploading(false);
      setSelectedFiles([]);
      if (created && fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).slice(0, MAX_UPLOADS);
    if (files.length === 0) {
      return;
    }

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
    setSelectedFiles([]);
    setPendingUploads((previous) => {
      revokePendingUploads(previous);
      return [];
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  const draftSelected = isDraftView && !currentTask;

  return (
    <main className="min-h-screen px-4 py-6 md:px-6">
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
        <aside className="min-h-[calc(100vh-3rem)] w-full max-w-[300px] shrink-0 border-r border-black/10 pr-5">
          <div className="sticky top-6 flex h-[calc(100vh-3rem)] flex-col">
            <div className="flex items-center justify-between px-3 pb-4">
              <p className="text-sm font-medium text-black/70">任务列表</p>
                <button
                  type="button"
                  onClick={openDraftTask}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-[12px] border border-black/10 bg-white/70 text-black/65 transition hover:bg-white"
                  aria-label="新建任务"
                >
                  <Plus size={17} strokeWidth={1.8} />
              </button>
            </div>

            <div className="space-y-1 overflow-y-auto pr-2">
              {draftSelected ? (
                <button
                  type="button"
                  onClick={openDraftTask}
                  className="w-full rounded-[14px] bg-white/75 px-3 py-3 text-left shadow-sm"
                >
                  <p className="truncate text-sm font-medium text-ink">新的测试任务</p>
                  <p className="mt-1 text-xs text-black/45">等待上传图片</p>
                </button>
              ) : null}

              {tasks.map((task) => {
                const active = !isDraftView && currentTask?.task_id === task.task_id;
                return (
                  <button
                    key={task.task_id}
                    type="button"
                    onClick={() => fetchTask(task.task_id).catch(() => null)}
                    className={`w-full rounded-[14px] px-3 py-3 text-left transition ${
                      active ? "bg-white/75 shadow-sm" : "hover:bg-white/45"
                    }`}
                  >
                    <p className="truncate text-sm font-medium text-ink">{taskDisplayLabel(task)}</p>
                    <p className="mt-1 truncate text-xs text-black/45">
                      {formatStatus(task.status)} · {formatTaskTime(task.updated_at) || formatStage(task.stage)}
                    </p>
                  </button>
                );
              })}

              {tasks.length === 0 && !draftSelected ? (
                <div className="rounded-2xl px-3 py-3 text-sm text-black/50">
                  还没有任务。点击右上角加号后开始上传图片。
                </div>
              ) : null}
            </div>
          </div>
        </aside>

          <section className="min-w-0 flex-1 space-y-6 pb-32 pl-8">
            {isDraftView ? (
            <>
              <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.92)] p-8 shadow-card">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">New Task</p>
                <h1 className="mt-4 font-display text-5xl leading-[1.06] tracking-tight text-ink md:text-6xl">新的测试任务</h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-black/62">
                  这里先保留一个等待上传图片的空页面。真正的后台任务会在你选择图片并完成上传后才创建，上传回调结束后会立刻启动人脸识别。
                </p>
              </section>

              <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">上传入口</p>
                    <p className="mt-2 text-base text-black/64">单次最多上传 {MAX_UPLOADS} 张图片。上传完成后会自动创建任务并进入人脸识别。</p>
                    {selectedFiles.length > 0 ? (
                      <p className="mt-3 font-mono text-xs uppercase tracking-[0.2em] text-black/42">已选择 {selectedFiles.length} / {MAX_UPLOADS}</p>
                    ) : null}
                  </div>

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="inline-flex items-center justify-center rounded-[14px] bg-[#1f1a15] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#2d251e] disabled:cursor-not-allowed disabled:bg-black/20"
                  >
                    选择图片并开始
                  </button>
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
            <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(250,246,239,0.92)] p-8 shadow-card">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">Current Task</p>
                  <h1 className="mt-4 font-display text-5xl leading-[1.06] tracking-tight text-ink md:text-6xl">{taskDisplayLabel(currentTask)}</h1>
                  <p className="mt-4 max-w-3xl text-base leading-7 text-black/62">
                    任务创建后会先进入人脸识别，再继续图片理解和后续推理。后面一组对话里你告诉我如何展示 boxed 人脸结果后，我会直接接到这块结果区。
                  </p>
                </div>

                <div className="rounded-[14px] border border-[#ddcebb] bg-white/70 px-5 py-4">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">状态</p>
                  <p className="mt-2 text-xl font-semibold">{formatStatus(currentTask.status)}</p>
                  <p className="mt-1 text-sm text-black/56">{formatStage(currentTask.stage)}</p>
                </div>
              </div>
            </section>
          ) : null}

          {galleryItems.length > 0 ? <UploadCarousel items={galleryItems} showRecognitionBadge={showRecognitionBadge} /> : null}

          {!isDraftView && (currentTask?.status === "running" || currentTask?.status === "queued") ? (
            <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-5 shadow-card">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="font-medium text-ink">任务正在处理中</p>
                  <p className="mt-1 text-sm text-black/58">当前阶段：{formatStage(currentTask.stage)}</p>
                </div>
                <WaitingDots label={`当前阶段：${formatStage(currentTask.stage)}`} />
              </div>
            </section>
          ) : null}

          {!isDraftView && stats.length > 0 ? (
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {stats.map((stat) => (
                <div key={stat.label} className="rounded-[16px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">{stat.label}</p>
                  <p className="mt-3 font-display text-5xl leading-none text-ink">{stat.value}</p>
                </div>
              ))}
            </section>
          ) : null}

          {!isDraftView && currentTask ? (
            <section className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
              <div className="space-y-6">
                <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                  <div className="flex items-center justify-between gap-4">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">任务概览</p>
                    <span className="rounded-[10px] bg-[#ece2d4] px-3 py-1 font-mono text-xs text-black/55">
                      {currentTask.task_id.slice(0, 8)}
                    </span>
                  </div>
                  <div className="mt-4 space-y-2 text-sm text-black/64">
                    <p>任务 ID：{currentTask.task_id}</p>
                    <p>当前阶段：{formatStage(currentTask.stage)}</p>
                    <p>提交图片数：{currentTask.upload_count}</p>
                    {currentTask.result?.summary.primary_person_id ? <p>主用户 ID：{currentTask.result.summary.primary_person_id}</p> : null}
                  </div>
                </section>

                {faceRecognition ? (
                  <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">人物汇总</p>
                    <p className="mt-3 font-display text-4xl text-ink">{faceRecognition.primary_person_id ?? "未识别主用户"}</p>
                    <div className="mt-4 space-y-3">
                      {(faceRecognition.persons ?? []).map((person) => (
                        <div key={person.person_id} className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4">
                          <div className="flex items-center justify-between gap-3">
                            <span className="font-medium">{person.person_id}</span>
                            <span className="font-mono text-xs text-black/50">{person.photo_count} 张照片</span>
                          </div>
                          <p className="mt-2 text-sm text-black/60">
                            人脸数：{person.face_count} · 平均得分：{person.avg_score?.toFixed?.(3) ?? person.avg_score}
                          </p>
                        </div>
                      ))}

                      {(faceRecognition.persons ?? []).length === 0 ? (
                        <div className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4 text-sm text-black/58">
                          当前任务尚未识别到人物，或人脸识别仍在进行中。
                        </div>
                      ) : null}
                    </div>
                  </section>
                ) : null}

                {warnings.length > 0 ? (
                  <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">任务警告</p>
                    <div className="mt-4 space-y-3">
                      {warnings.map((warning) => (
                        <div key={`${warning.stage}-${warning.message}`} className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4 text-sm text-black/65">
                          <p className="font-medium">{formatStage(warning.stage)}</p>
                          <p className="mt-1 leading-6">{warning.message}</p>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}

                <FailureList failures={failures} />
              </div>

              <div className="space-y-5">
                {imageEntries.length > 0 ? (
                  imageEntries.map((image) => {
                    const displayUrl = toAbsoluteUrl(image.display_image_url);
                    const boxedUrl = toAbsoluteUrl(image.boxed_image_url);
                    return (
                      <article
                        key={image.image_id}
                        className="overflow-hidden rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] shadow-card"
                      >
                        <div className="grid gap-0 lg:grid-cols-[1.08fr_0.92fr]">
                          <div className="bg-[#ece2d3]">
                            {displayUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img src={displayUrl} alt={image.filename} className="h-full w-full object-cover" />
                            ) : (
                              <div className="flex min-h-72 items-center justify-center text-sm text-black/38">暂无可预览图片</div>
                            )}
                          </div>

                          <div className="p-6">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-lg font-semibold">{image.filename}</span>
                              <span className="rounded-[10px] bg-black/5 px-2 py-1 font-mono text-xs">{image.image_id}</span>
                            </div>
                            <p className="mt-2 text-sm text-black/60">
                              检测到 {image.face_count} 张脸 {image.timestamp ? `· ${image.timestamp}` : ""}
                            </p>

                            {boxedUrl ? (
                              <a href={boxedUrl} target="_blank" rel="noreferrer" className="mt-3 inline-flex text-sm text-[#8a5637] underline">
                                查看 boxed image
                              </a>
                            ) : null}

                            <div className="mt-5 space-y-3">
                              {image.faces.length > 0 ? (
                                image.faces.map((face) => (
                                  <div key={face.face_id} className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="font-medium">{face.person_id}</span>
                                      <span className="rounded-[10px] bg-white px-2 py-1 font-mono text-xs">face {face.face_id.slice(0, 8)}</span>
                                    </div>
                                    <p className="mt-2 text-sm text-black/65">
                                      图片 ID：{face.image_id} · 分数 {face.score.toFixed(3)} · 相似度 {face.similarity.toFixed(3)}
                                    </p>
                                  </div>
                                ))
                              ) : (
                                <div className="rounded-[12px] border border-[#e1cfbf] bg-[#fbf5ed] p-4 text-sm text-black/58">
                                  这张图片没有识别到人脸。
                                </div>
                              )}

                              {image.failures && image.failures.length > 0 ? (
                                <div className="rounded-[12px] border border-[#e6cdbf] bg-[#fbf2ea] p-4">
                                  <p className="text-sm font-medium text-[#8a5637]">该图片后续处理存在失败记录</p>
                                  <div className="mt-2 space-y-2">
                                    {image.failures.map((failure) => (
                                      <p key={`${failure.image_id}-${failure.step}`} className="text-sm text-black/68">
                                        [{failure.step}] {failure.error}
                                      </p>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })
                ) : (
                  <section className="rounded-[18px] border border-[#d8c9b7] bg-[rgba(249,244,237,0.94)] p-8 shadow-card">
                    <p className="text-lg font-medium">还没有可展示的人脸识别结果</p>
                    <p className="mt-2 text-sm text-black/58">
                      下一组对话里你告诉我 boxed 人脸怎么排布后，我直接把结果视图接到这里。
                    </p>
                  </section>
                )}
              </div>
            </section>
          ) : null}

          <RecallChatDock />
        </section>
      </div>
    </main>
  );
}
