"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import type { FailureItem, TaskState } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const MAX_UPLOADS = 100;

function toAbsoluteUrl(url?: string | null) {
  if (!url) {
    return null;
  }
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  return `${API_BASE}${url}`;
}

function FailureList({ failures }: { failures: FailureItem[] }) {
  if (!failures.length) {
    return (
      <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
        <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">Bad Images</p>
        <p className="mt-3 text-sm text-black/70">No failed images recorded for this task.</p>
      </div>
    );
  }

  return (
    <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">Bad Images</p>
        <span className="rounded-full bg-ember/10 px-3 py-1 font-mono text-xs text-ember">
          {failures.length}
        </span>
      </div>
      <div className="mt-4 space-y-3">
        {failures.map((failure) => (
          <div key={`${failure.image_id}-${failure.step}`} className="rounded-2xl border border-ember/20 bg-ember/5 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold">{failure.filename}</span>
              <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.image_id}</span>
              <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{failure.step}</span>
            </div>
            <p className="mt-2 text-sm text-black/70">{failure.error}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [task, setTask] = useState<TaskState | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<number | null>(null);

  const faceRecognition = task?.result?.face_recognition;
  const imageEntries = faceRecognition?.images ?? [];
  const failures = task?.result?.failed_images ?? [];
  const canSubmit = files.length > 0 && files.length <= MAX_UPLOADS && !isSubmitting;

  useEffect(() => {
    if (!task || task.status === "completed" || task.status === "failed") {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    pollingRef.current = window.setInterval(async () => {
      const response = await fetch(`${API_BASE}/api/tasks/${task.task_id}`);
      if (!response.ok) {
        return;
      }
      const data = (await response.json()) as TaskState;
      setTask(data);
    }, 3000);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [task]);

  const stats = useMemo(() => {
    if (!task?.result) {
      return [];
    }

    return [
      { label: "Uploaded", value: task.result.summary.total_uploaded },
      { label: "Faces", value: task.result.summary.total_faces },
      { label: "Persons", value: task.result.summary.total_persons },
      { label: "Failed", value: task.result.summary.failed_images }
    ];
  }, [task]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []).slice(0, MAX_UPLOADS);
    setFiles(selected);
    setError(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    setIsSubmitting(true);
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
        throw new Error(payload?.detail ?? "Unable to create task");
      }

      const payload = (await response.json()) as { task_id: string };
      const taskResponse = await fetch(`${API_BASE}/api/tasks/${payload.task_id}`);
      const taskState = (await taskResponse.json()) as TaskState;
      setTask(taskState);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unknown error");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-8 md:px-8 md:py-12">
      <div className="mx-auto max-w-7xl space-y-8">
        <section className="overflow-hidden rounded-[2rem] border border-black/10 bg-[linear-gradient(135deg,rgba(255,255,255,0.92),rgba(255,244,232,0.72))] p-8 shadow-card md:p-12">
          <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <div>
              <p className="font-mono text-sm uppercase tracking-[0.25em] text-black/45">Memory Engineering</p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold tracking-tight text-ink md:text-6xl">
                Upload up to 100 photos and inspect face-recognition output first.
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-black/70 md:text-lg">
                This console submits images to the backend pipeline, polls task status, and renders boxed images,
                face IDs, person IDs, image IDs, plus a separate bad-image ledger for files that fail during processing.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="rounded-[1.75rem] border border-black/10 bg-white/75 p-6 shadow-card">
              <label className="block text-sm font-medium text-black/70">Photo batch</label>
              <div className="mt-3 rounded-3xl border border-dashed border-black/15 bg-sand/70 p-5">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileChange}
                  className="block w-full text-sm text-black/70 file:mr-4 file:rounded-full file:border-0 file:bg-ink file:px-4 file:py-2 file:font-medium file:text-white"
                />
                <p className="mt-3 font-mono text-xs uppercase tracking-[0.2em] text-black/45">
                  {files.length} / {MAX_UPLOADS} selected
                </p>
              </div>

              <button
                type="submit"
                disabled={!canSubmit}
                className="mt-5 inline-flex w-full items-center justify-center rounded-full bg-ink px-5 py-3 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:bg-black/20"
              >
                {isSubmitting ? "Creating task..." : "Start processing"}
              </button>

              {error ? <p className="mt-4 text-sm text-ember">{error}</p> : null}
              {task ? (
                <div className="mt-5 rounded-3xl border border-black/10 bg-black/[0.03] p-4">
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-black/45">Task Status</p>
                  <p className="mt-2 text-2xl font-semibold capitalize">{task.status}</p>
                  <p className="mt-1 text-sm text-black/60">Stage: {task.stage}</p>
                </div>
              ) : null}
            </form>
          </div>
        </section>

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

        {faceRecognition ? (
          <section className="grid gap-6 xl:grid-cols-[0.75fr_1.25fr]">
            <div className="space-y-6">
              <div className="rounded-3xl border border-black/10 bg-white/70 p-6 shadow-card">
                <p className="font-mono text-sm uppercase tracking-[0.2em] text-black/50">Primary Person</p>
                <p className="mt-3 text-3xl font-semibold">{faceRecognition.primary_person_id ?? "Unresolved"}</p>
                <div className="mt-4 space-y-3">
                  {(faceRecognition.persons ?? []).map((person) => (
                    <div key={person.person_id} className="rounded-2xl border border-black/10 bg-black/[0.03] p-4">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">{person.person_id}</span>
                        <span className="font-mono text-xs text-black/50">{person.photo_count} photos</span>
                      </div>
                      <p className="mt-2 text-sm text-black/60">Faces: {person.face_count} · Avg score: {person.avg_score?.toFixed?.(3) ?? person.avg_score}</p>
                    </div>
                  ))}
                </div>
              </div>

              <FailureList failures={failures} />
            </div>

            <div className="space-y-5">
              {imageEntries.map((image) => {
                const displayUrl = toAbsoluteUrl(image.display_image_url);
                return (
                  <article key={image.image_id} className="overflow-hidden rounded-3xl border border-black/10 bg-white/80 shadow-card">
                    <div className="grid gap-0 lg:grid-cols-[1.1fr_0.9fr]">
                      <div className="bg-ink/5">
                        {displayUrl ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={displayUrl} alt={image.filename} className="h-full w-full object-cover" />
                        ) : (
                          <div className="flex min-h-72 items-center justify-center bg-black/[0.04] text-sm text-black/45">
                            No preview available
                          </div>
                        )}
                      </div>
                      <div className="p-6">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-lg font-semibold">{image.filename}</span>
                          <span className="rounded-full bg-black/5 px-2 py-1 font-mono text-xs">{image.image_id}</span>
                        </div>
                        <p className="mt-2 text-sm text-black/60">
                          Faces detected: {image.face_count} {image.timestamp ? `· ${image.timestamp}` : ""}
                        </p>

                        <div className="mt-5 space-y-3">
                          {image.faces.length ? (
                            image.faces.map((face) => (
                              <div key={face.face_id} className="rounded-2xl border border-black/10 bg-sand/60 p-4">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="font-medium">{face.person_id}</span>
                                  <span className="rounded-full bg-white px-2 py-1 font-mono text-xs">face {face.face_id.slice(0, 8)}</span>
                                </div>
                                <p className="mt-2 text-sm text-black/65">
                                  Image ID: {face.image_id} · score {face.score.toFixed(3)} · similarity {face.similarity.toFixed(3)}
                                </p>
                              </div>
                            ))
                          ) : (
                            <div className="rounded-2xl border border-black/10 bg-black/[0.03] p-4 text-sm text-black/60">
                              No faces detected in this image.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        ) : null}
      </div>
    </main>
  );
}
