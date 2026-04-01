"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AlertTriangle, ArrowLeft, LoaderCircle } from "lucide-react";

import type { ReflectionTaskDetailResponse } from "@/lib/types";


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
      minute: "2-digit",
    }).format(new Date(value));
  } catch {
    return value;
  }
}


function humanizeOption(option?: string) {
  const mapping: Record<string, string> = {
    field_cot: "改 COT",
    tool_rule: "改 tool 规则",
    call_policy: "改 tool 找证据的索引",
    engineering_issue: "工程问题",
    watch_only: "继续观察先不改",
    critic_rule: "改 Critic 规则",
    judge_boundary: "改 Judge 边界",
    need_more_evidence: "先补更多证据",
    approve: "批准并执行",
    reject: "驳回",
    need_revision: "需要修订",
  };
  if (!option) {
    return "";
  }
  return mapping[option] ?? option;
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


export default function TaskReviewView({ taskId }: { taskId: string }) {
  const [payload, setPayload] = useState<ReflectionTaskDetailResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const response = await apiFetch(`${API_BASE}/api/reflection/tasks/${taskId}`);
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error((body && body.detail) || `加载失败 (${response.status})`);
        }
        const nextPayload = (await response.json()) as ReflectionTaskDetailResponse;
        if (!cancelled) {
          setPayload(nextPayload);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "加载反思任务失败");
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
  }, [taskId]);

  return (
    <main className="min-h-screen px-6 py-8 text-stone-900 md:px-10">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 flex items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-stone-500">Reflection Review</p>
            <h1 className="mt-2 font-[var(--font-newsreader)] text-4xl leading-tight text-stone-900">本地反思任务详情</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-stone-600">
              这里只展示离线聚类后的任务真源、support cases 和 evidence package，不提供审批操作。
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
              <span>正在加载反思任务...</span>
            </div>
          </section>
        ) : null}

        {!loading && error ? (
          <section className="rounded-[28px] border border-amber-300/70 bg-amber-50/90 p-8 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
            <div className="flex items-center gap-3 text-amber-900">
              <AlertTriangle className="h-5 w-5" />
              <div>
                <p className="font-semibold">无法加载这个任务</p>
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
                  <span className="rounded-full bg-stone-900 px-3 py-1 font-mono text-xs uppercase tracking-[0.24em] text-white">
                    {payload.task.task_type}
                  </span>
                  <span className="rounded-full bg-stone-200/80 px-3 py-1 text-xs uppercase tracking-[0.18em] text-stone-700">
                    {payload.task.lane}
                  </span>
                  <span className="rounded-full bg-orange-100 px-3 py-1 text-xs font-medium text-orange-900">
                    priority: {payload.task.priority}
                  </span>
                </div>
                <h2 className="mt-5 text-2xl font-semibold leading-tight text-stone-900">{payload.task.summary}</h2>
                <dl className="mt-6 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Task ID</dt>
                    <dd className="mt-2 font-mono text-sm text-stone-800">{payload.task.task_id}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Pattern ID</dt>
                    <dd className="mt-2 font-mono text-sm text-stone-800">{payload.pattern.pattern_id}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Recommended Option</dt>
                    <dd className="mt-2 text-sm font-semibold text-stone-900">{humanizeOption(payload.task.recommended_option)}</dd>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/75 p-4">
                    <dt className="text-xs uppercase tracking-[0.2em] text-stone-500">Current Status</dt>
                    <dd className="mt-2 text-sm font-semibold text-stone-900">{payload.task.status}</dd>
                  </div>
                </dl>
              </article>

              <aside className="rounded-[30px] border border-stone-300/70 bg-[rgba(247,243,236,0.95)] p-8 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Decision Context</p>
                <ul className="mt-5 space-y-4 text-sm text-stone-700">
                  <li className="rounded-2xl border border-stone-200/80 bg-white/70 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Created</p>
                    <p className="mt-2">{formatDateTime(payload.task.created_at)}</p>
                  </li>
                  <li className="rounded-2xl border border-stone-200/80 bg-white/70 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Updated</p>
                    <p className="mt-2">{formatDateTime(payload.task.updated_at)}</p>
                  </li>
                  <li className="rounded-2xl border border-stone-200/80 bg-white/70 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Options</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {payload.task.options.map((option) => (
                        <span key={option} className="rounded-full border border-stone-300/80 px-3 py-1 text-xs text-stone-700">
                          {humanizeOption(option)}
                        </span>
                      ))}
                    </div>
                  </li>
                </ul>
              </aside>
            </section>

            {payload.proposal ? (
              <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                <article className="rounded-[28px] border border-stone-300/70 bg-[rgba(236,246,255,0.92)] p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                  <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Proposal Review</p>
                  <h3 className="mt-4 text-xl font-semibold text-stone-900">{payload.proposal.summary}</h3>
                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                      <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Agent 判断依据</p>
                      <p className="mt-3 text-sm leading-6 text-stone-700">{payload.proposal.agent_reasoning_summary ?? "暂无"}</p>
                    </div>
                    <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                      <p className="text-xs uppercase tracking-[0.2em] text-stone-500">为什么不是其他改面</p>
                      <p className="mt-3 text-sm leading-6 text-stone-700">{payload.proposal.why_not_other_surfaces ?? "暂无"}</p>
                    </div>
                  </div>
                </article>
                <article className="rounded-[28px] border border-stone-300/70 bg-white/75 p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                  <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Experiment & Diff</p>
                  <div className="mt-5 grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-4">
                      <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Baseline Metrics</p>
                      <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-stone-700">
                        {renderValue(payload.proposal.experiment_report?.baseline_metrics ?? payload.proposal.baseline_metrics ?? {})}
                      </pre>
                    </div>
                    <div className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-4">
                      <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Candidate Metrics</p>
                      <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-stone-700">
                        {renderValue(payload.proposal.experiment_report?.candidate_metrics ?? payload.proposal.candidate_metrics ?? {})}
                      </pre>
                    </div>
                  </div>
                  <div className="mt-4 rounded-2xl border border-stone-200/80 bg-stone-50/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">具体修改内容</p>
                    <ul className="mt-3 space-y-2 text-sm text-stone-700">
                      {(payload.proposal.diff_summary ?? []).map((line, index) => (
                        <li key={`${line}-${index}`}>{line}</li>
                      ))}
                    </ul>
                  </div>
                </article>
              </section>
            ) : null}

            <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
              <article className="rounded-[28px] border border-stone-300/70 bg-white/75 p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Pattern Summary</p>
                <h3 className="mt-4 text-xl font-semibold text-stone-900">{payload.pattern.summary}</h3>
                <div className="mt-5 grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Root Causes</p>
                    <ul className="mt-3 space-y-2 text-sm text-stone-700">
                      {payload.pattern.root_cause_candidates.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Fix Surfaces</p>
                    <ul className="mt-3 space-y-2 text-sm text-stone-700">
                      {payload.pattern.fix_surface_candidates.map((item) => (
                        <li key={item}>{humanizeOption(item)}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </article>

              <article className="rounded-[28px] border border-stone-300/70 bg-[rgba(255,250,244,0.92)] p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Evidence Refs</p>
                <div className="mt-5 space-y-3">
                  {payload.evidence_refs.map((ref, index) => (
                    <div key={`${ref.source_type ?? "unknown"}-${ref.source_id ?? index}`} className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                      <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">
                        {ref.source_type || "unknown"} / {ref.source_id || "unknown"}
                      </p>
                      <p className="mt-2 text-sm text-stone-800">{ref.description || "无描述"}</p>
                      {ref.feature_names && ref.feature_names.length > 0 ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {ref.feature_names.map((feature) => (
                            <span key={feature} className="rounded-full bg-stone-900 px-3 py-1 text-[11px] text-white">
                              {feature}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              </article>
            </section>

            <section className="grid gap-6 xl:grid-cols-[0.78fr_1.22fr]">
              <article className="rounded-[28px] border border-stone-300/70 bg-white/75 p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Stage Trace</p>
                <div className="mt-5 space-y-3">
                  {payload.stage_trace.map((entry) => (
                    <div key={entry.case_id} className="rounded-2xl border border-stone-200/80 bg-stone-50/75 p-4">
                      <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">{entry.case_id}</p>
                      <p className="mt-2 text-sm text-stone-800">
                        {entry.first_seen_stage} → {entry.surfaced_stage}
                      </p>
                      <p className="mt-1 text-sm text-stone-600">{entry.signal_source}</p>
                      {entry.triage_reason ? <p className="mt-1 text-sm text-stone-500">{entry.triage_reason}</p> : null}
                    </div>
                  ))}
                </div>
              </article>

              <article className="rounded-[28px] border border-stone-300/70 bg-[rgba(247,243,236,0.95)] p-7 shadow-[0_24px_80px_rgba(56,43,32,0.08)]">
                <p className="font-mono text-xs uppercase tracking-[0.24em] text-stone-500">Support Cases</p>
                <div className="mt-5 grid gap-4 xl:grid-cols-3">
                  <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">GT Comparison</p>
                    <div className="mt-3 space-y-3">
                      {payload.gt_comparisons.map((item, index) => (
                        <div key={`${item.case_id ?? "comparison"}-${index}`} className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-3">
                          <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">{item.case_id ?? "comparison"}</p>
                          <p className="mt-2 text-sm text-stone-800">{item.grade ?? "unknown"} / {item.method ?? "unknown"}</p>
                          <p className="mt-1 text-xs text-stone-600">score: {item.score ?? 0}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Field Trace Summary</p>
                    <div className="mt-3 space-y-3">
                      {payload.field_trace_summaries.map((item) => (
                        <div key={item.case_id} className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-3">
                          <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">{item.case_id}</p>
                          <p className="mt-2 text-sm text-stone-800">{item.batch_name ?? "unknown batch"}</p>
                          <p className="mt-1 text-xs text-stone-600">
                            tool_called: {String(item.tool_called ?? false)} / retrieval_hit_count: {item.retrieval_hit_count ?? 0}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-stone-200/80 bg-white/80 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Route Decision</p>
                    <div className="mt-3 space-y-3">
                      {payload.route_decisions.map((item) => (
                        <div key={item.case_id} className="rounded-2xl border border-stone-200/80 bg-stone-50/80 p-3">
                          <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">{item.case_id}</p>
                          <p className="mt-2 text-sm text-stone-800">{item.resolution_route ?? "unrouted"}</p>
                          <p className="mt-1 text-xs text-stone-600">
                            {item.accuracy_gap_status ?? "unknown"} / {item.comparison_grade ?? "no_grade"}
                          </p>
                          <p className="mt-1 text-xs text-stone-600">
                            pre-audit: {item.pre_audit_comparison_grade ?? "unknown"} / causality: {item.causality_route ?? "none"}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mt-5 space-y-4">
                  {payload.support_cases.map((item) => (
                    <div key={item.case_id} className="rounded-3xl border border-stone-200/80 bg-white/80 p-5">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <p className="font-mono text-xs uppercase tracking-[0.2em] text-stone-500">{item.case_id}</p>
                          <h4 className="mt-2 text-lg font-semibold text-stone-900">{item.dimension}</h4>
                        </div>
                        <div className="text-right text-xs uppercase tracking-[0.2em] text-stone-500">
                          <div>{item.signal_source}</div>
                          <div className="mt-2">{item.business_priority}</div>
                        </div>
                      </div>
                      <div className="mt-4 grid gap-4 md:grid-cols-2">
                        <div className="rounded-2xl border border-stone-200/80 bg-stone-50/75 p-4">
                          <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Decision Trace</p>
                          <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-stone-700">
                            {renderValue(item.decision_trace ?? {})}
                          </pre>
                        </div>
                        <div className="rounded-2xl border border-stone-200/80 bg-stone-50/75 p-4">
                          <p className="text-xs uppercase tracking-[0.2em] text-stone-500">Output / Challenge</p>
                          <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-6 text-stone-700">
                            {renderValue({
                              upstream_output: item.upstream_output ?? {},
                              downstream_challenge: item.downstream_challenge ?? {},
                              downstream_judge: item.downstream_judge ?? {},
                            })}
                          </pre>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            </section>
          </div>
        ) : null}
      </div>
    </main>
  );
}
