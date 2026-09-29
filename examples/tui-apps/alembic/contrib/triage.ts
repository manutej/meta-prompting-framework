/**
 * Triage for the Ormus agent harness — TypeScript port of alembic's jev/triage.go.
 *
 * Deterministic score for every task on every tick (free), then ONE batched Jev call
 * for the ambiguous tasks under a budget declared before spending. Same scoring, same
 * record shape as the Go implementation, so alembic's history reads either.
 *
 *   import { Triager, loadFeed } from "./triage";
 *   const tr = new Triager({ apiKey: process.env.TYPESAFE_API_KEY });   // no key → deterministic only
 *   setInterval(async () => {
 *     const t = await tr.run(loadFeed("~/.ormus/feed.jsonl"), pendingPings);
 *     appendFileSync("~/.alembic/triage.jsonl", JSON.stringify(t) + "\n");
 *     const next = t.tasks.find(x => x.score > 0);   // what to work on next
 *   }, 60_000);
 *
 * Zero dependencies (Node 18+). Tests: node --experimental-strip-types --test triage.test.ts
 */
import { existsSync, readFileSync } from "node:fs";
import type { Task, TaskState, EventLevel } from "./ormus-adapter";

export const PRICE_PER_MILLION_INPUT = 0.042; // USD, TypeSafe list price; output is free

export type Action = "wait" | "ping" | "review" | "cancel" | "retry";

export interface Budget { maxTasksPerTick: number; maxCallsPerHour: number; chunkSize: number }
export const defaultBudget = (): Budget => ({ maxTasksPerTick: 8, maxCallsPerHour: 60, chunkSize: 6 });

export interface Event { task_id: string; level: EventLevel; text: string; at?: string }
export interface Snapshot { tasks: Record<string, Task & { updated: string }>; events: Record<string, Event[]> }

export interface TaskTriage {
  id: string; rank: number; score: number; base: number; reasons: string[];
  next: Action; next_conf: number; jev_used: boolean;
  stuck?: number; needs_human?: number; next_probs?: Record<string, number>;
}
export interface Triage {
  at: string; tasks: TaskTriage[]; jev_calls: number; jev_tasks: number; mock: boolean;
  usage: { input_tokens: number; output_tokens: number }; cost_usd: number; latency: number;
  skipped?: string; deterministic: boolean;
}

type Question = { type: "noul" | "choice"; instructions: string; criteria: Record<string, string> };
type Answer = { type: string; noul?: number; choice?: string; probabilities?: Record<string, number>; confidence?: number };
export interface Asker { ask(state: unknown, questions: Record<string, Question>): Promise<{ answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number }; mock: boolean }> }

const STATE_BASE: Record<TaskState, number> = { blocked: 0.75, failed: 0.7, review: 0.55, running: 0.35, queued: 0.2, done: 0 };
const DEFAULT_NEXT: Record<TaskState, Action> = { blocked: "ping", failed: "retry", review: "review", running: "wait", queued: "wait", done: "wait" };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const minutes = (ms: number) => Math.floor(ms / 60_000);
function shortDur(ms: number): string {
  const h = ms / 3_600_000;
  if (h < 1) return `${minutes(ms)}m`;
  if (h < 48) return `${Math.floor(h)}h`;
  return `${Math.floor(h / 24)}d`;
}
const plural = (n: number) => (n === 1 ? "" : "s");

/** Live TypeSafe client. Retries 429/529 with backoff; any other error propagates. */
export class JevClient implements Asker {
  private apiKey: string; private endpoint: string; private model: string;
  constructor(apiKey: string, endpoint = "https://api.typesafe.ai/v1/systemone", model = "jev-latest") {
    this.apiKey = apiKey; this.endpoint = endpoint; this.model = model;
  }
  async ask(state: unknown, questions: Record<string, Question>) {
    let last: Error | undefined;
    for (let attempt = 0; attempt < 3; attempt++) {
      const res = await fetch(this.endpoint, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, state, questions }),
      });
      if (res.ok) {
        const body = (await res.json()) as { answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } };
        return { answers: body.answers, usage: body.usage, mock: false };
      }
      last = new Error(`jev: HTTP ${res.status}`);
      if (res.status !== 429 && res.status !== 529) throw last;
      await new Promise(r => setTimeout(r, 500 * 2 ** attempt));
    }
    throw last;
  }
}

export class Triager {
  private calls: number[] = [];
  private base = new Map<string, { progress: number; since: number }>();
  readonly budget: Budget;
  readonly client?: Asker;
  now: () => number = Date.now;

  constructor(opts: { apiKey?: string; client?: Asker; budget?: Budget } = {}) {
    this.budget = opts.budget ?? defaultBudget();
    this.client = opts.client ?? (opts.apiKey ? new JevClient(opts.apiKey) : undefined);
  }

  /** Score one task with no model call. Mirrors Triager.Deterministic in Go. */
  deterministic(t: Task & { updated: string }, events: Event[] = [], pendingPings = 0): TaskTriage {
    const now = this.now();
    const tt: TaskTriage = { id: t.id, rank: 0, score: 0, base: 0, reasons: [t.state], next: DEFAULT_NEXT[t.state], next_conf: 0, jev_used: false };
    if (t.state === "done") return tt;
    let s = STATE_BASE[t.state];
    if ((t.priority ?? 0) > 0) { s += 0.1 * Math.min(2, t.priority!); tt.reasons.push(`priority ${t.priority}`); }
    const age = now - Date.parse(t.updated);
    if (t.state === "running" && age > 30 * 60_000) { s += 0.25; tt.reasons.push(`stale ${shortDur(age)}`); }
    else if (t.state === "running" && age > 10 * 60_000) { s += 0.15; tt.reasons.push(`stale ${shortDur(age)}`); }
    else if ((t.state === "blocked" || t.state === "review") && age > 30 * 60_000) { s += 0.1; tt.reasons.push(`waiting ${shortDur(age)}`); }
    let errs = 0, warns = 0;
    for (const e of events.slice(-10)) { if (e.level === "error") errs++; else if (e.level === "warn") warns++; }
    if (errs > 0) { s += Math.min(0.15, 0.05 * errs); tt.reasons.push(`${errs} error${plural(errs)}`); }
    if (warns > 0) s += Math.min(0.06, 0.02 * warns);
    if (pendingPings > 0) { s += 0.1; tt.reasons.push(`${pendingPings} ping${plural(pendingPings)} unanswered`); }
    const b = this.base.get(t.id);
    if (b && t.state === "running" && b.progress === (t.progress ?? 0) && now - b.since >= 5 * 60_000) {
      s += 0.1; tt.reasons.push(`no progress for ${shortDur(now - b.since)}`);
    }
    tt.base = clamp(s, 0, 1); tt.score = tt.base;
    return tt;
  }

  private ambiguous(t: Task, tt: TaskTriage): boolean {
    if (t.state === "done" || t.state === "queued") return false;
    if (t.state === "blocked" || t.state === "review" || t.state === "failed") return true;
    return tt.reasons.some(r => r.startsWith("stale") || r.endsWith("unanswered") || r.includes("error") || r.startsWith("no progress"));
  }

  private withinBudget(now: number): boolean {
    this.calls = this.calls.filter(c => c > now - 3_600_000);
    return this.budget.maxCallsPerHour <= 0 || this.calls.length < this.budget.maxCallsPerHour;
  }

  static qid(taskId: string, q: string): string { return taskId.replace(/[^A-Za-z0-9]/g, "_") + "__" + q; }

  /** The namespaced question set for a batch — identical wording to the Go side. */
  static questions(tasks: Task[]): Record<string, Question> {
    const qs: Record<string, Question> = {};
    for (const t of tasks) {
      const ref = `Task ${t.id} (${JSON.stringify(t.title)})`;
      qs[Triager.qid(t.id, "stuck")] = { type: "noul", instructions: `${ref}: the agent is looping or has stopped making progress.`,
        criteria: { true: "repeated identical attempts, or no forward movement for a long period", false: "events show forward movement" } };
      qs[Triager.qid(t.id, "needs_human")] = { type: "noul", instructions: `${ref}: a human decision is required before the agent can continue.`,
        criteria: { true: "waiting on approval, credentials, or a product decision", false: "the agent can proceed on its own" } };
      qs[Triager.qid(t.id, "next")] = { type: "choice", instructions: `${ref}: what should the operator do next?`,
        criteria: { wait: "nothing; the agent is progressing", ping: "send a short nudge or clarification", review: "open the output and review it now", cancel: "stop the task; it is off course", retry: "restart from the last good state" } };
    }
    return qs;
  }

  /** One tick. Never throws for Jev failures: the deterministic ranking is always produced. */
  async run(snap: Snapshot, pendingPings: Record<string, number> = {}): Promise<Triage> {
    const start = this.now();
    const out: Triage = { at: new Date(start).toISOString(), tasks: [], jev_calls: 0, jev_tasks: 0, mock: false,
      usage: { input_tokens: 0, output_tokens: 0 }, cost_usd: 0, latency: 0, deterministic: true };
    const ids = Object.keys(snap.tasks).sort();
    for (const id of ids) out.tasks.push(this.deterministic(snap.tasks[id], snap.events[id] ?? [], pendingPings[id] ?? 0));
    this.rank(out);

    let candidates = out.tasks.filter(tt => this.ambiguous(snap.tasks[tt.id], tt)).map(tt => tt.id);
    if (!this.client || this.budget.maxTasksPerTick <= 0) out.skipped = "jev disabled";
    else if (candidates.length === 0) out.skipped = "nothing ambiguous";
    else if (!this.withinBudget(start)) out.skipped = `budget: ${this.budget.maxCallsPerHour} calls/hour reached`;
    else {
      candidates = candidates.slice(0, this.budget.maxTasksPerTick);
      const chunk = this.budget.chunkSize > 0 ? this.budget.chunkSize : 6;
      for (let i = 0; i < candidates.length; i += chunk) {
        if (!this.withinBudget(this.now())) { out.skipped = "budget reached mid-tick"; break; }
        await this.askChunk(out, candidates.slice(i, i + chunk), snap);
      }
      if (out.jev_calls > 0) this.rank(out);
    }

    const next = new Map<string, { progress: number; since: number }>();
    for (const id of ids) {
      const p = snap.tasks[id].progress ?? 0;
      const b = this.base.get(id);
      next.set(id, b && b.progress === p ? b : { progress: p, since: start });
    }
    this.base = next;
    out.latency = this.now() - start;
    if (!out.mock) out.cost_usd = (out.usage.input_tokens / 1e6) * PRICE_PER_MILLION_INPUT;
    return out;
  }

  private async askChunk(out: Triage, ids: string[], snap: Snapshot): Promise<void> {
    const now = this.now();
    const state = ids.map(id => {
      const t = snap.tasks[id];
      return { id: t.id, title: t.title, state: t.state, agent: t.agent, progress: t.progress ?? 0, status_line: t.status_line,
        minutes_since_update: minutes(now - Date.parse(t.updated)), recent_events: (snap.events[id] ?? []).slice(-5).map(e => `${e.level}: ${e.text}`) };
    });
    this.calls.push(now);
    let resp;
    try { resp = await this.client!.ask(state, Triager.questions(ids.map(id => snap.tasks[id]))); }
    catch (e) { out.skipped = `jev error: ${(e as Error).message}`; return; }
    out.jev_calls++; out.mock = out.mock || resp.mock;
    out.usage.input_tokens += resp.usage.input_tokens; out.usage.output_tokens += resp.usage.output_tokens;
    for (const id of ids) {
      const tt = out.tasks.find(x => x.id === id); if (!tt) continue;
      const stuck = resp.answers[Triager.qid(id, "stuck")], human = resp.answers[Triager.qid(id, "needs_human")], next = resp.answers[Triager.qid(id, "next")];
      if (!stuck && !human && !next) continue;
      tt.jev_used = true; out.jev_tasks++; out.deterministic = false;
      let s = tt.base;
      if (stuck?.noul !== undefined) { tt.stuck = stuck.noul; s += 0.15 * tt.stuck; if (tt.stuck >= 0.7) tt.reasons.push(`jev: stuck ${tt.stuck.toFixed(2)}`); }
      if (human?.noul !== undefined) { tt.needs_human = human.noul; s += 0.15 * tt.needs_human; if (tt.needs_human >= 0.7) tt.reasons.push(`jev: needs human ${tt.needs_human.toFixed(2)}`); }
      if (next?.choice) { tt.next_probs = next.probabilities; const c = next.confidence ?? 0; if (c >= 0.6) { tt.next = next.choice as Action; tt.next_conf = c; } }
      tt.score = clamp(s, 0, 1);
    }
  }

  private rank(out: Triage): void {
    out.tasks.sort((a, b) => (b.score - a.score) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    out.tasks.forEach((t, i) => (t.rank = i + 1));
  }
}

/** Fold a feed.jsonl into a snapshot (same merge rules as harness/types.go). */
export function loadFeed(path: string): Snapshot {
  const snap: Snapshot = { tasks: {}, events: {} };
  if (!existsSync(path)) return snap;
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (!line.trim()) continue;
    let r: any; try { r = JSON.parse(line); } catch { continue; }
    if (r.type === "task.upsert" && r.task?.id) {
      const prev = snap.tasks[r.task.id];
      const t = { ...r.task, workflow: r.task.workflow ?? "default", updated: r.task.updated ?? r.ts };
      if (prev) { t.elements ??= prev.elements; t.created ??= prev.created; }
      snap.tasks[r.task.id] = t;
    } else if (r.type === "task.event" && r.event?.task_id) {
      const e = { ...r.event, at: r.event.at ?? r.ts };
      (snap.events[e.task_id] ??= []).push(e);
      if (snap.events[e.task_id].length > 500) snap.events[e.task_id].splice(0, snap.events[e.task_id].length - 500);
      const t = snap.tasks[e.task_id];
      if (t && e.level !== "info") { t.status_line = e.text; t.updated = e.at; }
    }
  }
  return snap;
}
