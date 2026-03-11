"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import type { FailureItem, TaskListResponse, TaskState } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_UPLOADS = 100;

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

function WaitingDots({ label = "处理中" }: { label?: string }) {
  return (
    <div className="inline-flex items-center gap-3 rounded-full bg-pine/10 px-4 py-2 text-sm text-pine">
      <svg width="44" height="14" viewBox="0 0 44 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.25;1;0.25" dur="1s" begin="0s" repeatCount="indefinite" />
        </circle>
        <circle cx="22" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.25;1;0.25" dur="1s" begin="0.18s" repeatCount="indefinite" />
        </circle>
        <circle cx="37" cy="7" r="5" fill="currentColor">
          <animate attributeName="opacity" values="0.25;1;0.25" dur="1s" begin="0.36s" repeatCount="indefinite" />
        </circle>
      </svg>
      <span>{label}</span>
    </div>
  );
}

function FailureList({ failures }: { failures: FailureItem[] }) {
  return (
    <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">坏图与失败记录</p>
        <span className="rounded-full bg-ember/10 px-3 py-1 font-mono text-xs text-ember">{failures.length}</span>
      </div>

      {failures.length === 0 ? (
        <p className="mt-3 text-sm text-black/70">当前任务没有记录到坏图或处理失败图片。</p>
      ) : (
        <div className="mt-4 space-y-3">
          {failures.map((failure) => (
            <div key={`${failure.image_id}-${failure.step}-${failure.filename}`} className="rounded-2xl border border-ember/20 bg-ember/5 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{failure.filename}</span>
                <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.image_id}</span>
                <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.step}</span>
              </div>
              <p className="mt-2 text-sm text-black/70">{failure.error}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const [tasks, setTasks] = useState<TaskState[]>([]);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const faceRecognition = currentTask?.result?.face_recognition;
  const imageEntries = faceRecognition?.images ?? [];
  const failures = currentTask?.result?.failed_images ?? [];
  const warnings = currentTask?.result?.warnings ?? [];

  const stats = useMemo(() => {
    if (!currentTask?.result) {
      return [];
    }

    return [
      { label: "已上传", value: currentTask.result.summary.total_uploaded },
      { label: "识别人脸", value: currentTask.result.summary.total_faces },
      { label: "识别人物", value: currentTask.result.summary.total_persons },
      { label: "失败图片", value: currentTask.result.summary.failed_images }
    ];
  }, [currentTask]);

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
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "创建任务失败");
    } finally {
      setIsUploading(false);
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).slice(0, MAX_UPLOADS);
    setSelectedFiles(files);

    if (files.length === 0) {
      return;
    }

    void createTask(files);
  }

  const currentUploads = currentTask?.uploads ?? [];

  return (
    <main className="min-h-screen px-4 py-6 md:px-6">
      <div className="mx-auto flex max-w-[1600px] gap-6">
        <aside className="sticky top-6 h-[calc(100vh-3rem)] w-full max-w-sm shrink-0 overflow-hidden rounded-[2rem] border border-black/10 bg-[linear-gradient(180deg,rgba(17,19,24,0.96),rgba(29,77,79,0.96))] text-white shadow-card">
          <div className="flex h-full flex-col">
            <div className="border-b border-white/10 px-6 py-6">
              <p className="font-mono text-xs uppercase tracking-[0.28em] text-white/45">测试任务面板</p>
              <h1 className="mt-3 text-3xl font-semibold leading-tight">上传照片后自动创建任务</h1>
              <p className="mt-3 text-sm leading-6 text-white/65">
                每次上传都会自动生成新的任务 ID，并把该批照片与任务绑定。当前单次最多支持 {MAX_UPLOADS} 张图片。
              </p>
            </div>

            <div className="border-b border-white/10 px-6 py-6">
              <label className="block text-sm font-medium text-white/80">上传测试照片</label>
              <div className="mt-3 rounded-[1.5rem] border border-dashed border-white/20 bg-white/5 p-5">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileChange}
                  disabled={isUploading}
                  className="block w-full text-sm text-white/75 file:mr-4 file:rounded-full file:border-0 file:bg-white file:px-4 file:py-2 file:font-medium file:text-ink"
                />
                <p className="mt-3 font-mono text-xs uppercase tracking-[0.22em] text-white/45">
                  已选择 {selectedFiles.length} / {MAX_UPLOADS}
                </p>
                <p className="mt-2 text-sm text-white/55">选完图片后将立即上传并创建任务，无需额外点击按钮。</p>
              </div>

              {isUploading ? (
                <div className="mt-4">
                  <WaitingDots label="正在上传并创建任务" />
                </div>
              ) : null}

              {error ? <p className="mt-4 text-sm text-[#ffb4a2]">{error}</p> : null}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
              <div className="mb-3 flex items-center justify-between px-2">
                <p className="font-mono text-xs uppercase tracking-[0.22em] text-white/45">最近任务</p>
                <button
                  type="button"
                  onClick={() => fetchTasks().catch(() => null)}
                  className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/70 transition hover:bg-white/10"
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
                          ? "border-white/40 bg-white/12"
                          : "border-white/10 bg-white/[0.04] hover:bg-white/[0.08]"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-mono text-xs text-white/55">{task.task_id.slice(0, 8)}</span>
                        <span className="rounded-full bg-white/10 px-2 py-1 text-xs text-white/70">
                          {formatStatus(task.status)}
                        </span>
                      </div>
                      <p className="mt-3 text-sm text-white/85">任务阶段：{formatStage(task.stage)}</p>
                      <p className="mt-1 text-xs text-white/55">图片数：{task.upload_count}</p>
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
          <div className="overflow-hidden rounded-[2rem] border border-black/10 bg-[linear-gradient(135deg,rgba(255,255,255,0.92),rgba(255,244,232,0.72))] p-8 shadow-card md:p-10">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="font-mono text-sm uppercase tracking-[0.25em] text-black/45">Memory Engineering Prototype</p>
                <h2 className="mt-4 text-4xl font-semibold tracking-tight text-ink md:text-5xl">中文测试任务工作台</h2>
                <p className="mt-4 max-w-3xl text-base leading-7 text-black/70">
                  右侧展示当前任务的人脸识别结果、boxed image、每张脸的 person_id / image_id，以及处理失败图片。
                  这套结构已经适合部署到 Railway 做测试联调用途。
                </p>
              </div>

              {currentTask ? (
                <div className="rounded-[1.5rem] border border-black/10 bg-white/70 px-5 py-4">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/45">当前任务</p>
                  <p className="mt-2 text-xl font-semibold">{currentTask.task_id}</p>
                  <p className="mt-1 text-sm text-black/60">
                    {formatStatus(currentTask.status)} · {formatStage(currentTask.stage)}
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          {currentTask?.status === "running" || currentTask?.status === "queued" ? (
            <div className="rounded-3xl border border-pine/15 bg-white/80 p-5 shadow-card">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="font-medium text-ink">任务正在处理中</p>
                  <p className="mt-1 text-sm text-black/60">当前阶段：{formatStage(currentTask.stage)}</p>
                </div>
                <WaitingDots label="请等待链路处理完成" />
              </div>
            </div>
          ) : null}

          {stats.length ? (
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {stats.map((stat) => (
                <div key={stat.label} className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/45">{stat.label}</p>
                  <p className="mt-3 text-4xl font-semibold">{stat.value}</p>
                </div>
              ))}
            </section>
          ) : null}

          {currentTask ? (
            <section className="grid gap-6 xl:grid-cols-[0.82fr_1.18fr]">
              <div className="space-y-6">
                <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                  <div className="flex items-center justify-between gap-4">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">任务概览</p>
                    <span className="rounded-full bg-black/5 px-3 py-1 font-mono text-xs text-black/55">
                      {formatStatus(currentTask.status)}
                    </span>
                  </div>
                  <div className="mt-4 space-y-2 text-sm text-black/70">
                    <p>任务 ID：{currentTask.task_id}</p>
                    <p>当前阶段：{formatStage(currentTask.stage)}</p>
                    <p>上传图片数：{currentTask.upload_count}</p>
                    {currentTask.result?.summary.primary_person_id ? (
                      <p>主用户 ID：{currentTask.result.summary.primary_person_id}</p>
                    ) : null}
                  </div>
                </div>

                <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                  <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">本任务上传图片</p>
                  <div className="mt-4 space-y-3">
                    {currentUploads.length > 0 ? (
                      currentUploads.map((upload) => (
                        <div key={upload.stored_filename} className="rounded-2xl border border-black/10 bg-black/[0.03] p-4">
                          <p className="font-medium">{upload.filename}</p>
                          <p className="mt-1 font-mono text-xs text-black/50">{upload.stored_filename}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-black/60">当前任务还没有记录上传列表。</p>
                    )}
                  </div>
                </div>

                {faceRecognition ? (
                  <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">人物汇总</p>
                    <p className="mt-3 text-3xl font-semibold">{faceRecognition.primary_person_id ?? "未识别主用户"}</p>
                    <div className="mt-4 space-y-3">
                      {(faceRecognition.persons ?? []).map((person) => (
                        <div key={person.person_id} className="rounded-2xl border border-black/10 bg-black/[0.03] p-4">
                          <div className="flex items-center justify-between gap-3">
                            <span className="font-medium">{person.person_id}</span>
                            <span className="font-mono text-xs text-black/50">{person.photo_count} 张照片</span>
                          </div>
                          <p className="mt-2 text-sm text-black/60">
                            人脸数：{person.face_count} · 平均得分：{person.avg_score?.toFixed?.(3) ?? person.avg_score}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <FailureList failures={failures} />

                {warnings.length > 0 ? (
                  <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                    <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">任务警告</p>
                    <div className="mt-4 space-y-3">
                      {warnings.map((warning) => (
                        <div key={`${warning.stage}-${warning.message}`} className="rounded-2xl border border-black/10 bg-black/[0.03] p-4 text-sm text-black/70">
                          <p className="font-medium">{formatStage(warning.stage)}</p>
                          <p className="mt-1">{warning.message}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="space-y-5">
                {imageEntries.length > 0 ? (
                  imageEntries.map((image) => {
                    const displayUrl = toAbsoluteUrl(image.display_image_url);
                    return (
                      <article key={image.image_id} className="overflow-hidden rounded-3xl border border-black/10 bg-white/80 shadow-card">
                        <div className="grid gap-0 lg:grid-cols-[1.08fr_0.92fr]">
                          <div className="bg-ink/5">
                            {displayUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img src={displayUrl} alt={image.filename} className="h-full w-full object-cover" />
                            ) : (
                              <div className="flex min-h-72 items-center justify-center bg-black/[0.04] text-sm text-black/45">
                                暂无可预览图片
                              </div>
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

                            <div className="mt-5 space-y-3">
                              {image.faces.length > 0 ? (
                                image.faces.map((face) => (
                                  <div key={face.face_id} className="rounded-2xl border border-black/10 bg-sand/60 p-4">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="font-medium">{face.person_id}</span>
                                      <span className="rounded-full bg-white px-2 py-1 font-mono text-xs">
                                        face {face.face_id.slice(0, 8)}
                                      </span>
                                    </div>
                                    <p className="mt-2 text-sm text-black/65">
                                      图片 ID：{face.image_id} · 分数 {face.score.toFixed(3)} · 相似度 {face.similarity.toFixed(3)}
                                    </p>
                                  </div>
                                ))
                              ) : (
                                <div className="rounded-2xl border border-black/10 bg-black/[0.03] p-4 text-sm text-black/60">
                                  这张图片没有识别到人脸。
                                </div>
                              )}

                              {image.failures && image.failures.length > 0 ? (
                                <div className="rounded-2xl border border-ember/20 bg-ember/5 p-4">
                                  <p className="text-sm font-medium text-ember">该图片后续处理存在失败记录</p>
                                  <div className="mt-2 space-y-2">
                                    {image.failures.map((failure) => (
                                      <p key={`${failure.image_id}-${failure.step}`} className="text-sm text-black/70">
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
                  <div className="rounded-3xl border border-black/10 bg-white/80 p-8 shadow-card">
                    <p className="text-lg font-medium">还没有可展示的图片结果</p>
                    <p className="mt-2 text-sm text-black/60">
                      从左侧面板上传图片后，这里会展示 boxed image、face 列表、person_id 和 image_id。
                    </p>
                  </div>
                )}
              </div>
            </section>
          ) : (
            <div className="rounded-3xl border border-black/10 bg-white/80 p-8 shadow-card">
              <p className="text-lg font-medium">请选择左侧任务，或直接上传一批图片开始新的测试任务。</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
