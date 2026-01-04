# Task List: MPG Core System

**Version:** 0.2.0-draft
**Status:** AWAITING APPROVAL
**Prerequisite:** plan.md approved

---

## Execution Strategy

**Approach**: MVP-first, TDD, incremental delivery

**Parallelization**: Tasks marked `[P]` can run in parallel with other `[P]` tasks in same phase.

**Estimation**: Each task estimated in hours. Total Phase 0: ~240 hours (8 weeks @ 30h/week)

---

## Phase 0: MVP (Weeks 1-8)

### Week 1-2: Foundation

#### Setup & Infrastructure

- [ ] **T0.1** `[P]` Initialize project structure (2h)
  - Create directories per plan.md structure
  - Setup package.json with dependencies
  - Configure TypeScript (tsconfig.json)
  - Configure Vitest
  - **Test**: `pnpm test` runs without error

- [ ] **T0.2** `[P]` Setup CI pipeline (2h)
  - GitHub Actions for test/lint/typecheck
  - Node.js 20.x matrix
  - **Test**: CI passes on empty project

- [ ] **T0.3** Create base types (4h)
  - Define Zod schemas for config
  - Generate TypeScript types from Zod
  - Export types from `src/types/`
  - **Test**: Types compile correctly

#### Config System

- [ ] **T0.4** Implement config loader (6h)
  - `loadConfig(path)` - read YAML file
  - `validateConfig(raw)` - Zod validation
  - `resolveDefaults(config)` - merge with defaults
  - **Test**: Valid YAML loads, invalid YAML errors with line number

- [ ] **T0.5** Implement env var substitution (3h)
  - Replace `${VAR}` patterns with process.env values
  - Error on missing required vars
  - **Test**: `${HOME}` substitutes correctly

- [ ] **T0.6** `[P]` Create test fixtures (2h)
  - `minimal.yaml` - simplest valid config
  - `full.yaml` - all options
  - `invalid/*.yaml` - error cases
  - **Test**: Fixtures match schema

**Week 1-2 Checkpoint**: Config loads and validates

---

### Week 3-4: Core Engine

#### Template System

- [ ] **T0.7** Create next-marketing template (8h)
  - Base Next.js 14 app
  - Tailwind + shadcn/ui setup
  - Variable placeholders: `{{site.name}}`, `{{brand.palette}}`
  - Layout components: hero, features, cta
  - **Test**: Template builds standalone

- [ ] **T0.8** Implement scaffold function (6h)
  - Copy template directory to output
  - Substitute variables in all files
  - Rename pattern files (`__name__.tsx`)
  - **Test**: Generated site matches expected structure

- [ ] **T0.9** `[P]` Implement layout expansion (4h)
  - Parse `hero+features+cta` notation
  - Map to component imports
  - Generate page file with components
  - **Test**: `hero+cta` expands to correct JSX

#### Orchestrator

- [ ] **T0.10** Implement job queue (6h)
  - `createJobs(sites, steps)` - create job objects
  - `executeJobs(jobs, concurrency)` - p-limit wrapper
  - Job states: pending → running → completed/failed
  - **Test**: 5 jobs with concurrency=2 executes correctly

- [ ] **T0.11** Implement progress tracking (4h)
  - Per-site status updates
  - Overall progress percentage
  - Time estimates
  - **Test**: Progress updates emit correctly

- [ ] **T0.12** Implement error recovery (4h)
  - Failed sites don't block others
  - Collect all errors for final report
  - Partial success is valid outcome
  - **Test**: 1 failing site, 4 succeed

**Week 3-4 Checkpoint**: Can scaffold 5 sites in parallel

---

### Week 5-6: CLI Interface

#### Commands

- [ ] **T0.13** Implement CLI skeleton (4h)
  - Commander.js setup
  - Subcommand structure
  - Help text generation
  - Version flag
  - **Test**: `mpg --help` shows all commands

- [ ] **T0.14** Implement `list` command (4h)
  - Load config
  - Display sites as table
  - Support `--type` filter
  - Support `--format=json`
  - **Test**: `mpg list type=marketing` filters correctly

- [ ] **T0.15** Implement `plan` command (4h)
  - Load config
  - Show what would be generated (dry run)
  - List affected files
  - Show step sequence
  - **Test**: `mpg plan --dry` matches actual apply

- [ ] **T0.16** Implement `apply` command (8h)
  - Load config
  - Create jobs
  - Execute with progress output
  - Summary on completion
  - **Test**: `mpg apply steps=scaffold` generates sites

- [ ] **T0.17** Implement `status` command (3h)
  - Show in-progress jobs
  - Show completed/failed counts
  - **Test**: Status updates during execution

- [ ] **T0.18** `[P]` Implement `view` command (3h)
  - Show single site details
  - List pages and layouts
  - **Test**: `mpg view site=alpha` shows config

#### Output Formatting

- [ ] **T0.19** Implement table formatter (3h)
  - Aligned columns
  - Color coding (success/error)
  - Truncation for long values
  - **Test**: Output is readable

- [ ] **T0.20** `[P]` Implement JSON formatter (2h)
  - Valid JSON output
  - Pretty-print option
  - **Test**: Output parses as JSON

**Week 5-6 Checkpoint**: Full CLI operational

---

### Week 7-8: MCP Integration & Polish

#### MCP Server

- [ ] **T0.21** Setup MCP server skeleton (4h)
  - Initialize with @modelcontextprotocol/sdk
  - Tool registration structure
  - Error handling wrapper
  - **Test**: Server starts without error

- [ ] **T0.22** Implement `list_sites` tool (3h)
  - Map to core.listSites()
  - JSON response format
  - **Test**: MCP client receives sites

- [ ] **T0.23** Implement `plan_sites` tool (3h)
  - Map to core.planSites()
  - Include affected files
  - **Test**: Dry run via MCP

- [ ] **T0.24** Implement `apply_sites` tool (4h)
  - Map to core.applySites()
  - Stream progress updates
  - **Test**: Generation via MCP

- [ ] **T0.25** `[P]` Write MCP integration tests (4h)
  - End-to-end with mock client
  - Error scenarios
  - **Test**: All tools callable

#### Documentation & Polish

- [ ] **T0.26** Create README.md (4h)
  - Quick start guide
  - Installation instructions
  - Command reference
  - Example sites.yaml
  - **Test**: New user can follow guide

- [ ] **T0.27** Create next-docs template (6h)
  - Fumadocs or similar
  - Sidebar layout
  - Search placeholder
  - **Test**: Template builds standalone

- [ ] **T0.28** Error message polish (3h)
  - Actionable suggestions
  - Color coding
  - Link to docs
  - **Test**: Errors are helpful

- [ ] **T0.29** Performance testing (4h)
  - Benchmark 10-site generation
  - Memory profiling
  - Identify bottlenecks
  - **Test**: Meets <60s target

- [ ] **T0.30** Security audit (3h)
  - Check for secret leaks
  - Validate path handling
  - Dependency audit
  - **Test**: No critical vulnerabilities

**Week 7-8 Checkpoint**: MVP complete, ready for beta

---

## Phase 0 Summary

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 1-2 | Foundation | Config system works |
| 3-4 | Core Engine | Parallel scaffolding works |
| 5-6 | CLI | All 5 commands work |
| 7-8 | MCP + Polish | Full system operational |

**Total Tasks**: 30
**Total Hours**: ~240h
**Parallel Opportunities**: 8 tasks

---

## Phase 1: Value-Add (Weeks 9-14) - DEFERRED

*To be planned after Phase 0 ships and user feedback collected*

Planned features:
- [ ] Deployment integration (Vercel, Netlify)
- [ ] Additional templates (Remix, Astro)
- [ ] Named workflow execution
- [ ] Incremental generation
- [ ] `run` command

---

## Phase 2: Pro Features (Weeks 15-22) - DEFERRED

*To be planned based on user demand*

Potential features:
- [ ] Visual builder sync (Builder.io)
- [ ] Multi-plane view system
- [ ] Advanced workflows (fork/wait-all)
- [ ] AI content generation

---

## Task Dependencies

```
T0.1 ──┬── T0.3 ── T0.4 ── T0.5 ── T0.7 ── T0.8 ── T0.10 ── T0.13...
       │
T0.2 ──┘
       │
T0.6 ──┴── (parallel)
       │
T0.9 ──┴── (parallel with T0.8)
```

---

## Risk Mitigation Tasks

| Risk | Mitigation Task |
|------|-----------------|
| Template complexity | T0.7 done early, iterate |
| MCP instability | T0.21-24 can be cut if blocked |
| Performance issues | T0.29 identifies problems early |

---

## Approval Checklist

- [ ] Every P1 requirement maps to a task
- [ ] Dependencies are explicit
- [ ] Parallel tasks marked `[P]`
- [ ] Time estimates are realistic (×1.5 buffer)
- [ ] Test coverage defined per task
- [ ] Phases are clearly separated

---

*Begin implementation only after this task list is approved.*
