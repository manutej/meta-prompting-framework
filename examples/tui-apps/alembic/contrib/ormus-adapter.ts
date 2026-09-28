/**
 * alembic adapter for an agent harness. Zero dependencies (Node 18+).
 *
 *   import { AlembicFeed, AlembicOutbox } from "./ormus-adapter";
 *   const feed = new AlembicFeed();                 // ~/.ormus/feed.jsonl
 *   feed.workflow({ id: "checkout", name: "checkout-service", env: "prod", state: "healthy" });
 *   feed.agent({ id: "coder-1", name: "Coder", model: "claude-opus-5-5", state: "busy" });
 *   feed.task({ id: "T-1041", workflow: "checkout", title: "…", agent: "coder-1",
 *               state: "running", progress: 0.55, status_line: "writing middleware" });
 *   feed.event("T-1041", "warn", "CI red on push 3");
 *
 *   const outbox = new AlembicOutbox();
 *   outbox.watch(cmd => {
 *     if (cmd.type === "ping") { deliverToAgent(cmd.agent, cmd.text); feed.ack(cmd.id, cmd.agent, "ack"); }
 *     if (cmd.type === "task.cancel") cancel(cmd.task_id);
 *   });
 *
 * Contract: docs/HARNESS-CONTRACT.md (v1). Both files are append-only.
 */
import { appendFileSync, existsSync, mkdirSync, openSync, readSync, statSync, closeSync, watchFile } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

export type TaskState = "queued" | "running" | "blocked" | "review" | "done" | "failed";
export type ElementKind = "file" | "pr" | "url" | "log" | "dir";
export type EventLevel = "info" | "warn" | "error" | "ok";

export interface Element { kind: ElementKind; ref: string; label?: string; line?: number }
export interface Task {
  id: string; workflow?: string; title: string; agent?: string; state: TaskState;
  progress?: number; status_line: string; worktree?: string; branch?: string;
  elements?: Element[]; priority?: 0 | 1 | 2; created?: string; updated?: string;
}
export interface Agent { id: string; name: string; model?: string; state: "idle" | "busy" | "offline"; current_task?: string; seen?: string }
export interface Workflow { id: string; name: string; env?: "prod" | "staging" | "dev"; state?: string }

export type Command =
  | { v: 1; id: string; ts: string; type: "ping"; agent: string; task_id?: string; text: string }
  | { v: 1; id: string; ts: string; type: "task.cancel" | "task.retry"; task_id: string }
  | { v: 1; id: string; ts: string; type: "jev.receipt"; task_id?: string; data: unknown };

const defaultDir = () => process.env.ORMUS_HOME ?? join(homedir(), ".ormus");

function appendLine(path: string, obj: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, JSON.stringify(obj) + "\n");
}

export class AlembicFeed {
  constructor(readonly path: string = process.env.ALEMBIC_FEED ?? join(defaultDir(), "feed.jsonl")) {}
  private write(type: string, payload: Record<string, unknown>): void {
    appendLine(this.path, { v: 1, ts: new Date().toISOString(), type, ...payload });
  }
  task(task: Task): void { this.write("task.upsert", { task: { updated: new Date().toISOString(), ...task } }); }
  event(task_id: string, level: EventLevel, text: string): void { this.write("task.event", { event: { task_id, level, text, at: new Date().toISOString() } }); }
  agent(agent: Agent): void { this.write("agent.upsert", { agent: { seen: new Date().toISOString(), ...agent } }); }
  workflow(workflow: Workflow): void { this.write("workflow.upsert", { workflow }); }
  ack(ping_id: string, agent: string, text = "ack"): void { this.write("ping.ack", { ping_ack: { ping_id, agent, text, at: new Date().toISOString() } }); }
}

export class AlembicOutbox {
  private offset = 0;
  private seen = new Set<string>();
  constructor(readonly path: string = process.env.ALEMBIC_OUTBOX ?? join(defaultDir(), "outbox.jsonl")) {}

  /** Read commands appended since the last call. Duplicate ids are dropped (one intent, one effect). */
  poll(): Command[] {
    if (!existsSync(this.path)) return [];
    const size = statSync(this.path).size;
    if (size < this.offset) this.offset = 0;
    if (size === this.offset) return [];
    const fd = openSync(this.path, "r");
    const buf = Buffer.alloc(size - this.offset);
    readSync(fd, buf, 0, buf.length, this.offset);
    closeSync(fd);
    const text = buf.toString("utf8");
    const lastNl = text.lastIndexOf("\n");
    if (lastNl < 0) return [];
    this.offset += Buffer.byteLength(text.slice(0, lastNl + 1));
    const out: Command[] = [];
    for (const line of text.slice(0, lastNl).split("\n")) {
      if (!line.trim()) continue;
      try {
        const cmd = JSON.parse(line) as Command;
        if (cmd.v !== 1 || !cmd.id || this.seen.has(cmd.id)) continue;
        this.seen.add(cmd.id);
        out.push(cmd);
      } catch { /* one bad line never poisons the outbox */ }
    }
    return out;
  }

  /** Deliver each new command to fn; polls on file change and every intervalMs. */
  watch(fn: (cmd: Command) => void, intervalMs = 500): () => void {
    const tick = () => { for (const c of this.poll()) fn(c); };
    tick();
    const timer = setInterval(tick, intervalMs);
    watchFile(this.path, { interval: intervalMs }, tick);
    return () => clearInterval(timer);
  }
}
