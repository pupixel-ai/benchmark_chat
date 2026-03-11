"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
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
    ? "border-black/10 bg-black/5 text-black/45"
    : "border-[#cdbba4] bg-[#f4ecdf] text-[#6f5847]";

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 ${
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
  title,
  description,
  items,
  showRecognitionBadge
}: {
  title: string;
  description: string;
  items: GalleryCard[];
  showRecognitionBadge: boolean;
}) {
  return (
    <section className="rounded-[2rem] border border-[#d6c7b3] bg-[rgba(250,246,239,0.88)] p-6 shadow-card">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/40">{title}</p>
          <p className="mt-2 text-sm leading-6 text-black/60">{description}</p>
        </div>
        <div className="text-sm text-black/45">横向滚动查看全部图片</div>
      </div>

      <div className="mt-5 flex snap-x gap-4 overflow-x-auto pb-2">
        {items.map((item) => (
          <article
            key={item.id}
            className="relative min-w-[260px] snap-start overflow-hidden rounded-[1.6rem] border border-[#d8c8b5] bg-[#f7f0e6]"
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
              {item.meta ? <p className="font-mono text-xs uppercase tracking-[0.14em] text-black/40">{item.meta}</p> : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function FailureList({ failures }: { failures: FailureItem[] }) {
  return (
    <div className="rounded-[2rem] border border-[#d8c6b2] bg-[rgba(248,243,236,0.9)] p-6 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">坏图与失败记录</p>
        <span className="rounded-full bg-[#ead8ca] px-3 py-1 font-mono text-xs text-[#8a5637]">{failures.length}</span>
      </div>

      {failures.length === 0 ? (
        <p className="mt-3 text-sm text-black/65">当前任务没有记录到坏图或处理失败图片。</p>
      ) : (
        <div className="mt-4 space-y-3">
          {failures.map((failure) => (
            <div
              key={`${failure.image_id}-${failure.step}-${failure.filename}`}
              className="rounded-[1.4rem] border border-[#e2cdbf] bg-[#fbf5ed] p-4"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{failure.filename}</span>
                <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.image_id}</span>
                <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.step}</span>
              </div>
              <p className="mt-2 text-sm leading-6 text-black/65">{failure.error}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RecallChatPlaceholder() {
  return (
    <section className="rounded-[2rem] border border-[#d6c7b3] bg-[rgba(242,236,228,0.84)] p-6 opacity-90 shadow-card">
      <div className="rounded-[1.8rem] border border-dashed border-black/10 bg-[rgba(255,255,255,0.34)] p-6">
        <p className="font-mono text-xs uppercase tracking-[0.24em] text-black/35">Recall Chat</p>
        <h3 className="mt-3 font-display text-3xl text-ink">召回测试聊天窗口</h3>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-black/50">
          这里会承接记忆布局完成后的对话式召回测试。目前界面保持灰置，用于锁定最终交互位置和尺寸。
        </p>

        <div className="mt-5 rounded-[1.6rem] border border-black/8 bg-[rgba(255,255,255,0.45)] p-4">
          <WaitingDots label="聊天能力暂未开放" muted />
        </div>

        <div className="mt-4 flex items-end gap-3 rounded-[1.6rem] border border-black/8 bg-[rgba(249,246,240,0.95)] p-4">
          <textarea
            disabled
            rows={3}
            placeholder="记忆布局完成后可在这里输入召回问题"
            className="min-h-[96px] flex-1 resize-none rounded-[1.2rem] border border-black/8 bg-[#f2eee8] px-4 py-3 text-sm text-black/40 outline-none"
          />
          <button
            type="button"
            disabled
            className="rounded-full border border-black/10 bg-[#ebe4d8] px-5 py-3 text-sm text-black/35"
          >
            发送
          </button>
        </div>
      </div>

      <p className="mt-4 px-1 text-xs text-black/45">记忆布局完成后可以进行召回测试</p>
    </section>
  );
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

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [tasks, setTasks] = useState<TaskState[]>([]);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
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

  async function fetchTasks(preserveCurrent = true) {
    const response = await fetch(`${API_BASE}/api/tasks?limit=30`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("获取任务列表失败");
    }
    const payload = (await response.json()) as TaskListResponse;
    setTasks(payload.tasks);

    if (!preserveCurrent && payload.tasks.length > 0) {
      setCurrentTask(payload.tasks[0]);
      return;
    }

    if (preserveCurrent && currentTask) {
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
    setTasks((previous) => {
      const next = previous.filter((task) => task.task_id !== payload.task_id);
      return [payload, ...next];
    });
  }

  useEffect(() => {
    fetchTasks(false).catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : "初始化任务列表失败");
    });
  }, []);

  useEffect(() => {
    if (!currentTask || currentTask.status === "completed" || currentTask.status === "failed") {
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
  }, [currentTask]);

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

  return (
    <main className="min-h-screen px-4 py-6 md:px-6">
      <div className="mx-auto flex max-w-[1640px] gap-6">
        <aside className="sticky top-6 h-[calc(100vh-3rem)] w-full max-w-sm shrink-0 overflow-hidden rounded-[2.2rem] border border-[#d4c2ad] bg-[linear-gradient(180deg,#24211d,#4d4035)] text-white shadow-card">
          <div className="flex h-full flex-col">
            <div className="border-b border-white/10 px-6 py-6">
              <p className="font-mono text-xs uppercase tracking-[0.28em] text-white/45">测试任务面板</p>
              <h1 className="mt-3 font-display text-[2.3rem] leading-tight text-[#f7f0e8]">上传照片后自动创建任务</h1>
              <p className="mt-3 text-sm leading-6 text-white/62">
                每次上传都会生成新的任务 ID，并把这一批图片统一绑定到任务下。服务端会在落盘时把图片转换成 WebP。
              </p>
            </div>

            <div className="border-b border-white/10 px-6 py-6">
              <label className="block text-sm font-medium text-white/80">上传测试照片</label>
              <div className="mt-3 rounded-[1.6rem] border border-dashed border-white/15 bg-white/5 p-5">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileChange}
                  disabled={isUploading}
                  className="block w-full text-sm text-white/70 file:mr-4 file:rounded-full file:border-0 file:bg-[#f7efe4] file:px-4 file:py-2 file:font-medium file:text-[#2d241d]"
                />
                <p className="mt-3 font-mono text-xs uppercase tracking-[0.22em] text-white/42">
                  已选择 {selectedFiles.length} / {MAX_UPLOADS}
                </p>
                <p className="mt-2 text-sm text-white/55">选完图片后立即上传，无需额外点击按钮。</p>
              </div>

              {isUploading ? (
                <div className="mt-4">
                  <WaitingDots label="正在上传并创建任务" />
                </div>
              ) : null}

              {error ? <p className="mt-4 text-sm text-[#ffcab5]">{error}</p> : null}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
              <div className="mb-3 flex items-center justify-between px-2">
                <p className="font-mono text-xs uppercase tracking-[0.22em] text-white/42">最近任务</p>
                <button
                  type="button"
                  onClick={() => fetchTasks().catch(() => null)}
                  className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/68 transition hover:bg-white/10"
                >
                  刷新
                </button>
              </div>

              <div className="space-y-3">
                {tasks.map((task) => {
                  const active = currentTask?.task_id === task.task_id;
                  return (
                    <button
                      key={task.task_id}
                      type="button"
                      onClick={() => fetchTask(task.task_id).catch(() => null)}
                      className={`w-full rounded-[1.4rem] border p-4 text-left transition ${
                        active
                          ? "border-[#e6d7c3] bg-white/14"
                          : "border-white/10 bg-white/[0.04] hover:bg-white/[0.08]"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-mono text-xs text-white/54">{task.task_id.slice(0, 8)}</span>
                        <span className="rounded-full bg-white/10 px-2 py-1 text-xs text-white/72">
                          {formatStatus(task.status)}
                        </span>
                      </div>
                      <p className="mt-3 text-sm text-white/88">阶段：{formatStage(task.stage)}</p>
                      <p className="mt-1 text-xs text-white/52">图片数：{task.upload_count}</p>
                    </button>
                  );
                })}

                {tasks.length === 0 ? (
                  <div className="rounded-[1.4rem] border border-white/10 bg-white/[0.04] p-4 text-sm text-white/60">
                    暂无任务。上传一批图片后会自动出现在这里。
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </aside>

        <section className="min-w-0 flex-1 space-y-6">
          <div className="overflow-hidden rounded-[2.2rem] border border-[#d6c7b3] bg-[linear-gradient(135deg,rgba(251,247,241,0.94),rgba(240,230,216,0.92))] p-8 shadow-card md:p-10">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="font-mono text-sm uppercase tracking-[0.25em] text-black/42">Memory Engineering Prototype</p>
                <h2 className="mt-4 max-w-4xl font-display text-5xl leading-[1.05] tracking-tight text-ink md:text-6xl">
                  更接近 Anthropic 气质的中文测试工作台
                </h2>
                <p className="mt-4 max-w-3xl text-base leading-7 text-black/65">
                  这里集中展示任务状态、人脸识别结果、boxed image、坏图记录和即将启用的召回测试入口，适合部署到 Railway 供测试同学直接联调。
                </p>
              </div>

              {currentTask ? (
                <div className="rounded-[1.6rem] border border-[#dcccb8] bg-white/70 px-5 py-4">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">当前任务</p>
                  <p className="mt-2 text-xl font-semibold">{currentTask.task_id}</p>
                  <p className="mt-1 text-sm text-black/58">
                    {formatStatus(currentTask.status)} · {formatStage(currentTask.stage)}
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          {currentTask?.status === "running" || currentTask?.status === "queued" ? (
            <div className="rounded-[2rem] border border-[#d9c9b6] bg-[rgba(249,244,237,0.92)] p-5 shadow-card">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="font-medium text-ink">任务正在处理中</p>
                  <p className="mt-1 text-sm text-black/58">当前阶段：{formatStage(currentTask.stage)}</p>
                </div>
                <WaitingDots label={`当前阶段：${formatStage(currentTask.stage)}`} />
              </div>
            </div>
          ) : null}

          {stats.length > 0 ? (
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {stats.map((stat) => (
                <div key={stat.label} className="rounded-[1.8rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-6 shadow-card">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/42">{stat.label}</p>
                  <p className="mt-3 font-display text-5xl leading-none text-ink">{stat.value}</p>
                </div>
              ))}
            </section>
          ) : null}

          {galleryItems.length > 0 ? (
            <UploadCarousel
              title={pendingUploads.length > 0 ? "刚提交的图片走廊" : "任务图片走廊"}
              description={
                pendingUploads.length > 0
                  ? "本地预览会在任务创建成功前先显示在这里，方便测试同学确认这批图片已经提交。"
                  : "这里保留任务原始上传图片的缩略画廊，文件名和 WebP 存储结果会一并展示。"
              }
              items={galleryItems}
              showRecognitionBadge={showRecognitionBadge}
            />
          ) : (
            <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-8 shadow-card">
              <p className="text-lg font-medium">上传图片后，这里会出现横向 gallery / carousel。</p>
              <p className="mt-2 text-sm text-black/58">缩略图、文件名，以及上传后的人脸识别等待态都会保留在这里。</p>
            </div>
          )}

          {currentTask ? (
            <section className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
              <div className="space-y-6">
                <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-6 shadow-card">
                  <div className="flex items-center justify-between gap-4">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">任务概览</p>
                    <span className="rounded-full bg-[#ece2d4] px-3 py-1 font-mono text-xs text-black/55">
                      {formatStatus(currentTask.status)}
                    </span>
                  </div>
                  <div className="mt-4 space-y-2 text-sm text-black/65">
                    <p>任务 ID：{currentTask.task_id}</p>
                    <p>当前阶段：{formatStage(currentTask.stage)}</p>
                    <p>提交图片数：{currentTask.upload_count}</p>
                    {currentTask.result?.summary.primary_person_id ? <p>主用户 ID：{currentTask.result.summary.primary_person_id}</p> : null}
                  </div>
                </div>

                {faceRecognition ? (
                  <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">人物汇总</p>
                    <p className="mt-3 font-display text-4xl text-ink">{faceRecognition.primary_person_id ?? "未识别主用户"}</p>
                    <div className="mt-4 space-y-3">
                      {(faceRecognition.persons ?? []).map((person) => (
                        <div key={person.person_id} className="rounded-[1.4rem] border border-[#dfd0bc] bg-[#fbf6ee] p-4">
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
                        <div className="rounded-[1.4rem] border border-[#dfd0bc] bg-[#fbf6ee] p-4 text-sm text-black/58">
                          当前任务尚未识别到人物，或人脸识别仍在进行中。
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {warnings.length > 0 ? (
                  <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/45">任务警告</p>
                    <div className="mt-4 space-y-3">
                      {warnings.map((warning) => (
                        <div key={`${warning.stage}-${warning.message}`} className="rounded-[1.4rem] border border-[#dfd0bc] bg-[#fbf6ee] p-4 text-sm text-black/65">
                          <p className="font-medium">{formatStage(warning.stage)}</p>
                          <p className="mt-1 leading-6">{warning.message}</p>
                        </div>
                      ))}
                    </div>
                  </div>
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
                        className="overflow-hidden rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] shadow-card"
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
                              <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{image.image_id}</span>
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
                                  <div key={face.face_id} className="rounded-[1.4rem] border border-[#dfd0bc] bg-[#fbf6ee] p-4">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="font-medium">{face.person_id}</span>
                                      <span className="rounded-full bg-white px-2 py-1 font-mono text-xs">face {face.face_id.slice(0, 8)}</span>
                                    </div>
                                    <p className="mt-2 text-sm text-black/65">
                                      图片 ID：{face.image_id} · 分数 {face.score.toFixed(3)} · 相似度 {face.similarity.toFixed(3)}
                                    </p>
                                  </div>
                                ))
                              ) : (
                                <div className="rounded-[1.4rem] border border-[#dfd0bc] bg-[#fbf6ee] p-4 text-sm text-black/58">
                                  这张图片没有识别到人脸。
                                </div>
                              )}

                              {image.failures && image.failures.length > 0 ? (
                                <div className="rounded-[1.4rem] border border-[#e6cdbf] bg-[#fbf2ea] p-4">
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
                  <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-8 shadow-card">
                    <p className="text-lg font-medium">还没有可展示的人脸识别结果</p>
                    <p className="mt-2 text-sm text-black/58">
                      上传图片后，这里会展示 face 列表、person_id、image_id，以及带框图片。
                    </p>
                  </div>
                )}
              </div>
            </section>
          ) : (
            <div className="rounded-[2rem] border border-[#d8cab8] bg-[rgba(249,244,237,0.92)] p-8 shadow-card">
              <p className="font-display text-3xl text-ink">请选择左侧任务，或直接上传一批图片开始新的测试任务。</p>
            </div>
          )}

          <RecallChatPlaceholder />
        </section>
      </div>
    </main>
  );
}
