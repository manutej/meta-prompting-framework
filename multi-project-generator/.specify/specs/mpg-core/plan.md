# Implementation Plan: MPG Core System

**Version:** 0.2.0-draft
**Status:** AWAITING APPROVAL
**Prerequisite:** spec.md approved

---

## Executive Summary

This plan translates the MPG specification into architectural decisions, technical context, and implementation contracts. It answers: "HOW do we build this?"

**Scope**: Phase 0 (MVP) only - 8-10 weeks, 1 developer

---

## 1. Technical Context

### 1.1 Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | TypeScript 5.5+ | Type safety, ecosystem |
| Runtime | Node.js 20+ | ESM support, async/await |
| Package Manager | pnpm | Fast, disk-efficient, monorepo support |
| Schema | Zod | Runtime validation + TypeScript inference |
| CLI Framework | Commander.js | Battle-tested, simple API |
| MCP SDK | @modelcontextprotocol/sdk | Official SDK |
| Parallel Execution | p-limit | Proven, minimal |
| YAML Parser | js-yaml | Standard library |
| Testing | Vitest | Fast, ESM-native |
| Output | Turborepo | Monorepo orchestration |

### 1.2 Platform Targets

| Platform | Support Level | Notes |
|----------|---------------|-------|
| macOS (ARM) | Full | Primary development platform |
| macOS (Intel) | Full | CI testing |
| Linux (x64) | Full | CI/CD, Docker |
| Windows (WSL) | Full | Via WSL2 only |
| Windows (native) | Limited | Path handling differences |

### 1.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start | <2s | Time to first command output |
| 10-site scaffold | <60s | End-to-end generation |
| Memory (100 sites) | <2GB | Peak RSS |
| MCP response | <500ms | Tool call latency |

---

## 2. Constitution Check

**Does this plan violate any constitutional articles?**

| Article | Check | Status |
|---------|-------|--------|
| I. Independent Testability | Each component testable in isolation | PASS |
| II. Explicit Assumptions | ADRs document decisions | PASS |
| III. Honest Timelines | 8-10 weeks, 1 developer | PASS |
| IV. MVP First | Scope limited to P1 stories | PASS |
| V. No Vendor Lock-In | No required external services | PASS |
| VI. Security by Design | Env var substitution for secrets | PASS |
| VII. Complexity Budget | 3 core components | PASS |

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: Monorepo Output Structure

**Problem**: Should generated sites live in one repo (monorepo) or separate repos (polyrepo)?

**Decision**: Monorepo using Turborepo

**Rationale**:
- Shared dependencies reduce disk usage
- Atomic updates across sites
- Unified CI/CD pipeline
- Turborepo handles parallel builds efficiently

**Tradeoffs**:
- Large repos can be slow to clone
- Permission management more complex
- Single point of failure for CI

**Alternatives Considered**:
- Polyrepo: More flexible but coordination overhead
- No orchestration: Too manual for 10+ sites

---

### ADR-002: Zod for Schema Validation

**Problem**: How to validate YAML configuration?

**Decision**: Use Zod schemas with TypeScript inference

**Rationale**:
- Single source of truth for types and validation
- Better error messages than JSON Schema alone
- Runtime validation + compile-time types
- Ecosystem support (trpc, etc.)

**Tradeoffs**:
- Additional dependency
- Schema must be maintained alongside types

**Alternatives Considered**:
- JSON Schema only: No TypeScript inference
- io-ts: More complex API
- Yup: Less TypeScript-native

---

### ADR-003: p-limit for Concurrency

**Problem**: How to manage parallel execution with limits?

**Decision**: Use p-limit library

**Rationale**:
- Minimal API (one function)
- Proven at scale (50M+ downloads/week)
- Easy to test and reason about
- No complex queue management needed for MVP

**Tradeoffs**:
- No persistence (jobs lost on crash)
- No distributed execution

**Alternatives Considered**:
- Bull/BullMQ: Overkill for MVP, requires Redis
- p-queue: More features but more complexity
- Custom implementation: NIH risk

---

### ADR-004: Commander.js for CLI

**Problem**: Which CLI framework to use?

**Decision**: Commander.js

**Rationale**:
- Most popular Node.js CLI framework
- Simple, declarative API
- Built-in help generation
- Extensible subcommand structure

**Tradeoffs**:
- Less opinionated than alternatives
- No built-in config file support

**Alternatives Considered**:
- oclif: Too heavy for MVP
- yargs: More complex API
- citty: Newer, less proven

---

### ADR-005: Template as File Copy + Substitution

**Problem**: How to implement template scaffolding?

**Decision**: Copy template directory, then substitute variables in files

**Rationale**:
- Simple to understand and debug
- No special template syntax needed
- Works with any file type
- Easy to add new templates

**Implementation**:
1. Copy template directory to output
2. Walk all files
3. Replace `{{variable}}` patterns
4. Rename files with patterns (e.g., `__name__.tsx` → `alpha.tsx`)

**Tradeoffs**:
- Not as powerful as full template engines
- Variable syntax visible in templates

**Alternatives Considered**:
- EJS/Handlebars: More complex, overkill for MVP
- Plop: Good but separate tool
- Hygen: Another dependency

---

## 4. Project Structure

```
multi-project-generator/
├── .specify/                    # Spec-kit structure
│   ├── memory/
│   │   └── constitution.md
│   ├── specs/mpg-core/
│   │   ├── spec.md
│   │   ├── plan.md (this file)
│   │   └── tasks.md
│   └── templates/
│
├── src/
│   ├── index.ts                 # Library entry point
│   ├── cli/
│   │   ├── index.ts             # CLI entry point
│   │   ├── commands/
│   │   │   ├── list.ts
│   │   │   ├── plan.ts
│   │   │   ├── apply.ts
│   │   │   ├── status.ts
│   │   │   └── view.ts
│   │   └── output.ts            # Formatters (table, json)
│   │
│   ├── core/
│   │   ├── config/
│   │   │   ├── loader.ts        # YAML loading
│   │   │   ├── schema.ts        # Zod schemas
│   │   │   └── resolver.ts      # Defaults + env vars
│   │   ├── orchestrator/
│   │   │   ├── index.ts         # Job queue manager
│   │   │   ├── executor.ts      # Step execution
│   │   │   └── progress.ts      # Status tracking
│   │   └── generator/
│   │       ├── scaffold.ts      # Template copying
│   │       ├── substitute.ts    # Variable replacement
│   │       └── layout.ts        # Layout expansion
│   │
│   ├── mcp/
│   │   ├── server.ts            # MCP server setup
│   │   └── tools/
│   │       ├── list-sites.ts
│   │       ├── plan-sites.ts
│   │       └── apply-sites.ts
│   │
│   └── types/
│       ├── config.ts            # Generated from Zod
│       └── job.ts               # Execution state
│
├── templates/
│   ├── next-marketing/
│   │   ├── template.yaml        # Template metadata
│   │   ├── package.json
│   │   ├── next.config.js
│   │   └── ...
│   └── next-docs/
│       └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── examples/
│   └── sites.yaml               # Example configuration
│
├── package.json
├── tsconfig.json
└── vitest.config.ts
```

---

## 5. Component Responsibilities

### 5.1 Config Loader (`src/core/config/`)

**Responsibility**: Load and validate YAML configuration

**Input**: File path to sites.yaml
**Output**: Typed `MPGConfig` object

**Key Functions**:
```typescript
loadConfig(path: string): Promise<MPGConfig>
validateConfig(raw: unknown): Result<MPGConfig, ValidationError[]>
resolveDefaults(config: MPGConfig): MPGConfig
substituteEnvVars(config: MPGConfig): MPGConfig
```

### 5.2 Orchestrator (`src/core/orchestrator/`)

**Responsibility**: Manage parallel execution of site generation

**Input**: List of sites, workflow steps, concurrency limit
**Output**: Execution results with status per site

**Key Functions**:
```typescript
createJobs(sites: SiteConfig[], steps: Step[]): Job[]
executeJobs(jobs: Job[], concurrency: number): AsyncGenerator<JobUpdate>
getStatus(): OrchestratorState
cancelJobs(jobIds: string[]): void
```

### 5.3 Generator (`src/core/generator/`)

**Responsibility**: Scaffold sites from templates

**Input**: Site config, template name, output path
**Output**: Generated files in output directory

**Key Functions**:
```typescript
scaffold(site: SiteConfig, template: string, outDir: string): Promise<void>
substituteVariables(content: string, vars: Record<string, unknown>): string
expandLayout(layout: string): ComponentTree
```

### 5.4 CLI (`src/cli/`)

**Responsibility**: Parse commands, invoke core, format output

**Input**: Command line arguments
**Output**: Formatted terminal output

**Commands**:
- `/mpg list` → `listCommand()`
- `/mpg plan` → `planCommand()`
- `/mpg apply` → `applyCommand()`
- `/mpg status` → `statusCommand()`
- `/mpg view` → `viewCommand()`

### 5.5 MCP Server (`src/mcp/`)

**Responsibility**: Expose core functionality as MCP tools

**Tools**:
```typescript
{
  name: "list_sites",
  inputSchema: { type?: SiteType },
  handler: async (input) => core.listSites(input)
}
```

---

## 6. API Contracts

### 6.1 CLI Commands

```bash
# List sites
mpg list [sites] [--type=<type>] [--format=<table|json>]

# Plan (dry run)
mpg plan [sites] [--type=<type>] [--steps=<step+step>] [--dry]

# Apply (execute)
mpg apply [sites] [--type=<type>] [--steps=<step+step>] [--concurrency=<n>]

# Status
mpg status [--job=<id>]

# View
mpg view [site=<id>] [--format=<table|json>]
```

### 6.2 MCP Tools

```typescript
// list_sites
{
  name: "list_sites",
  description: "List all configured sites",
  inputSchema: {
    type: "object",
    properties: {
      type: { type: "string", enum: ["marketing", "docs", "app", "landing"] },
      format: { type: "string", enum: ["summary", "full"] }
    }
  }
}

// plan_sites
{
  name: "plan_sites",
  description: "Preview generation (dry run)",
  inputSchema: {
    type: "object",
    properties: {
      ids: { type: "array", items: { type: "string" } },
      type: { type: "string" },
      steps: { type: "array", items: { type: "string" } }
    }
  }
}

// apply_sites
{
  name: "apply_sites",
  description: "Execute generation steps",
  inputSchema: {
    type: "object",
    required: ["steps"],
    properties: {
      ids: { type: "array", items: { type: "string" } },
      type: { type: "string" },
      steps: { type: "array", items: { type: "string" } },
      concurrency: { type: "number", minimum: 1, maximum: 50 }
    }
  }
}
```

---

## 7. Error Handling Strategy

### 7.1 Error Categories

| Category | Example | Response |
|----------|---------|----------|
| Config Error | Invalid YAML syntax | Abort with line number |
| Validation Error | Missing required field | Abort with field path |
| Template Error | Template not found | Abort with template name |
| Generation Error | File write failure | Log error, continue other sites |
| Network Error | MCP timeout | Retry with backoff |

### 7.2 Error Format

```typescript
interface MPGError {
  code: string;           // E.g., "CONFIG_INVALID"
  message: string;        // Human-readable
  path?: string;          // Config path (e.g., "sites[0].brand.palette")
  suggestion?: string;    // How to fix
  recoverable: boolean;   // Can execution continue?
}
```

---

## 8. Testing Strategy

### 8.1 Test Levels

| Level | Coverage Target | Focus |
|-------|-----------------|-------|
| Unit | 80% | Core logic, schema validation |
| Integration | Key paths | CLI → Core → Output |
| E2E | Happy path | Full generation workflow |

### 8.2 Test Fixtures

```
tests/fixtures/
├── valid/
│   ├── minimal.yaml       # Simplest valid config
│   ├── full.yaml          # All options used
│   └── 10-sites.yaml      # Scale test
├── invalid/
│   ├── missing-id.yaml
│   ├── bad-type.yaml
│   └── syntax-error.yaml
└── templates/
    └── test-template/     # Minimal template for tests
```

---

## 9. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Secrets in config | Require `${ENV_VAR}` syntax, never store literals |
| Secrets in logs | Redact patterns matching `sk-`, `key=`, etc. |
| Path traversal | Validate output paths are within project |
| Arbitrary code execution | No eval(), no shell interpolation |
| Supply chain | Pin dependencies, audit regularly |

---

## 10. Open Technical Questions

| Question | Impact | Resolution Path |
|----------|--------|-----------------|
| Node.js 18 vs 20? | Affects ESM support | Benchmark both |
| pnpm vs npm? | Monorepo performance | Test with 50 sites |
| Turbo vs Nx? | Output orchestration | Evaluate both |

---

## 11. Plan Approval Checklist

- [ ] ADRs document all major decisions
- [ ] Component responsibilities are clear
- [ ] API contracts are complete
- [ ] Error handling is defined
- [ ] Testing strategy is realistic
- [ ] Security considerations addressed
- [ ] Constitution check passes

---

*Proceed to tasks.md only after plan approval.*
