"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AlertTriangle, ArrowLeft, LoaderCircle } from "lucide-react";

import type { ReflectionDifficultCaseDetailResponse } from "@/lib/types";


const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");


async function apiFetch(input: string, init?: RequestInit) {
  return fetch(input, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.headers ?? {}),
    },
  });
}


function renderValue(value: unknown) {
  if (value == null) {
    return "null";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value, null, 2);
}


function renderCaseValue(value: unknown) {
  if (value == null || value === "") {
    return "空值 / 未识别";
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => String(item)).join("、") : "空值 / 未识别";
  }
  if (typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}


export default function DifficultCaseReviewView({ caseId }: { caseId: string }) {
  const [payload, setPayload] = useState<ReflectionDifficultCaseDetailResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const response = await apiFetch(`${API_BASE}/api/reflection/difficult-cases/${caseId}`);
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error((body && body.detail) || `加载失败 (${response.status})`);
        }
        const nextPayload = (await response.json()) as ReflectionDifficultCaseDetailResponse;
        if (!cancelled) {
          setPayload(nextPayload);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "加载疑难 case 失败");
          setPayload(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [caseId]);

  return (
    <main className="min-h-screen px-6 py-8 text-stone-900 md:px-10">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 flex items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-stone-500">Difficult Case Review</p>
            <h1 className="mt-2 font-[var(--font-newsreader)] text-4xl leading-tight text-stone-900">疑难 Case 详情</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-stone-600">
              这里直接展示 GT 标注、当前识别结果，以及为什么它暂时还难以稳定归因。
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex items-center gap-2 rounded-full border border-stone-300/80 bg-white/70 px-4 py-2 text-sm text-stone-700 shadow-sm transition hover:border-stone-400 hover:bg-white"
          >
            <ArrowLeft className="h-4 w-4" />
            返回工作台
          </Link>
        </div>

        {loading ? (
          <section className="rounded-[28px] border border-stone-300/70 bg-white/70 p-10 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
            <div className="flex items-center gap-3 text-stone-600">
              <LoaderCircle className="h-5 w-5 animate-spin" />
              <span>正在加载疑难 case...</span>
            </div>
          </section>
        ) : null}

        {!loading && error ? (
          <section className="rounded-[28px] border border-amber-300/70 bg-amber-50/90 p-8 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
            <div className="flex items-center gap-3 text-amber-900">
              <AlertTriangle className="h-5 w-5" />
              <div>
                <p className="font-semibold">无法加载这个疑难 case</p>
                <p className="mt-1 text-sm text-amber-800">{error}</p>
              </div>
            </div>
          </section>
        ) : null}

        {!loading && !error && payload ? (
          <div className="space-y-6">
            <section className="grid gap-6 lg:grid-cols-[1.35fr_0.85fr]">
              <article className="rounded-[30px] border border-stone-300/70 bg-[rgba(255,250,244,0.92)] p-8 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="rounded-full bg-red-700 px-3 py-1 font-mono text-xs uppercase tracking-[0.24em] text-white">
                    difficult_case
                  </span>
                  <span className="rounded-full bg-stone-200/80 px-3 py-1 text-xs uppercase tracking-[0.18em] text-stone-700">
                    {payload.route_decision.resolution_route ?? "unrouted"}
                  </span>
                </div>
                <h2 className="mt-5 text-2xl font-semibold leading-tight text-stone-900">{payload.case.summary}</h2>
                <dl className="mt-6 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Case ID</dt>
                    <dd className="mt-2 font-mono text-sm text-stone-800">{payload.case.case_id}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Dimension</dt>
                    <dd className="mt-2 text-sm text-stone-800">{payload.case.dimension}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Accuracy Gap</dt>
                    <dd className="mt-2 text-sm font-semibold text-stone-900">{payload.route_decision.accuracy_gap_status ?? "unknown"}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Causality Route</dt>
                    <dd className="mt-2 text-sm font-semibold text-stone-900">{payload.route_decision.causality_route ?? "unknown"}</dd>
                  </div>
                </dl>
              </article>

              <aside className="rounded-[30px] border border-stone-300/70 bg-[rgba(247,243,236,0.95)] p-8 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">GT Comparison</p>
                <dl className="mt-4 space-y-4">
                  <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">GT 标注</dt>
                    <dd className="mt-2 whitespace-pre-wrap text-sm leading-6 text-stone-800">
                      {renderCaseValue(payload.gt_comparison.gt_value)}
                    </dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">当前识别</dt>
                    <dd className="mt-2 whitespace-pre-wrap text-sm leading-6 text-stone-800">
                      {renderCaseValue(payload.gt_comparison.output_value)}
                    </dd>
                  </div>
                </dl>
              </aside>
            </section>

            <section>
              <article className="rounded-[28px] border border-stone-300/70 bg-[rgba(255,250,244,0.92)] p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Why It Is Hard</p>
                <div className="mt-4 rounded-2xl border border-stone-200/80 bg-white/80 p-5 text-sm leading-7 text-stone-800">
                  {payload.difficulty_reason ? renderValue(payload.difficulty_reason) : "当前归因还不稳定，需要进一步人工判断。"}
                </div>
              </article>
            </section>
          </div>
        ) : null}
      </div>
    </main>
  );
}
