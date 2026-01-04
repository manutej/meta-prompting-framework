# Multi-Project Site Generator (MPG)
## Comprehensive Specification Document

**Version:** 0.1.0-draft
**Status:** AWAITING APPROVAL
**Last Updated:** 2026-01-04

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [User Personas](#4-user-personas)
5. [Core Requirements](#5-core-requirements)
6. [Architecture Overview](#6-architecture-overview)
7. [Data Models](#7-data-models)
8. [DSL Specification](#8-dsl-specification)
9. [CLI Command Grammar](#9-cli-command-grammar)
10. [MCP Tool Interfaces](#10-mcp-tool-interfaces)
11. [Multi-Plane View System](#11-multi-plane-view-system)
12. [Visual Builder Integration](#12-visual-builder-integration)
13. [Orchestration Engine](#13-orchestration-engine)
14. [Template System](#14-template-system)
15. [Security Considerations](#15-security-considerations)
16. [Open Questions](#16-open-questions)
17. [Approval Checklist](#17-approval-checklist)

---

## 1. Executive Summary

### What is MPG?

The **Multi-Project Site Generator (MPG)** is an orchestration framework that enables:

- **Parallel generation** of 100+ websites from a single declarative configuration
- **Terminal-first** workflow with repeatable, scriptable architecture
- **MCP server integration** for agentic coding tools (Claude Code, etc.)
- **Visual builder bridges** (Builder.io, Plasmic) for drag-and-drop customization
- **Multi-plane views** showing the same system at different abstraction levels
- **Modern stack support** (React, Next.js, Remix, Astro, TypeScript)

### Core Value Proposition

```
┌─────────────────────────────────────────────────────────────────┐
│  ONE YAML CONFIG  →  100 SITES  →  PARALLEL GENERATION         │
│                                                                 │
│  sites.yaml ──→ MPG Orchestrator ──→ Claude Code Agents        │
│                       │                      │                  │
│                       ↓                      ↓                  │
│              Turborepo Monorepo    →   Deployed Sites          │
│                       │                                         │
│                       ↓                                         │
│              Visual Builder Sync (Builder.io / Plasmic)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Problem Statement

### Current Pain Points

| Problem | Impact | Frequency |
|---------|--------|-----------|
| Creating 100 similar sites requires 100x manual effort | High | Every project |
| No unified way to manage site variations at scale | High | Weekly |
| Visual builders don't integrate with terminal/code workflows | Medium | Daily |
| No way to view a project portfolio at different abstraction levels | Medium | Daily |
| MCP servers exist but aren't orchestrated for multi-site generation | High | Emerging |

### Who Has This Problem?

- **Agencies** building many client sites with similar structure
- **SaaS companies** creating per-tenant marketing sites
- **Enterprise** teams managing portfolio of internal tools
- **Indie hackers** launching multiple products simultaneously
- **DevRel teams** generating SDK documentation sites

---

## 3. Goals & Non-Goals

### Goals (In Scope)

| ID | Goal | Priority | Success Metric |
|----|------|----------|----------------|
| G1 | Define 100+ sites in a single YAML config | P0 | Config validates without errors |
| G2 | Generate sites in parallel (10-50 concurrent) | P0 | 100 sites in <10 minutes |
| G3 | Expose MCP tools for Claude Code integration | P0 | All verbs callable via MCP |
| G4 | Support visual builder sync (Builder.io) | P1 | Bidirectional sync works |
| G5 | Multi-plane views (business/pages/components/pipeline) | P1 | All 4 planes render correctly |
| G6 | Turborepo monorepo output structure | P1 | `turbo build` succeeds |
| G7 | Template system for site scaffolding | P1 | 5+ templates available |
| G8 | CLI with intuitive verb grammar | P0 | 10 core commands work |

### Non-Goals (Out of Scope for v0.1)

| ID | Non-Goal | Reason |
|----|----------|--------|
| NG1 | Production hosting management | Use Vercel/Netlify directly |
| NG2 | Full CMS implementation | Integrate with existing CMS |
| NG3 | Custom component library creation | Use existing UI kits |
| NG4 | Real-time collaborative editing | Future version |
| NG5 | GraphQL API generation | REST/tRPC only for now |
| NG6 | Mobile app generation | Web only |

---

## 4. User Personas

### Persona 1: Agency Developer ("Alex")

```yaml
name: Alex
role: Full-stack developer at digital agency
experience: 5 years
daily_tools: [VS Code, Terminal, Figma, Vercel]
pain_points:
  - "I build 3-5 client sites per month with similar structures"
  - "Each site takes 2-3 days to scaffold and configure"
  - "Clients want visual editing but I need code control"
needs:
  - Batch generate sites from templates
  - Visual builder integration for client handoff
  - Terminal-first workflow
success: "Generate 10 client sites in an afternoon"
```

### Persona 2: Platform Engineer ("Jordan")

```yaml
name: Jordan
role: Platform engineer at SaaS company
experience: 8 years
daily_tools: [Terminal, GitHub, Terraform, Nx]
pain_points:
  - "We have 50+ tenant marketing pages with slight variations"
  - "Changes need to propagate across all sites"
  - "No single view of our site portfolio"
needs:
  - Centralized configuration with overrides
  - Parallel deployment orchestration
  - Multi-plane visibility
success: "Update branding across 50 sites in one command"
```

### Persona 3: AI-Augmented Builder ("Sam")

```yaml
name: Sam
role: Indie hacker using Claude Code
experience: 2 years
daily_tools: [Claude Code, Terminal, Cursor]
pain_points:
  - "I want Claude to build my sites but need structure"
  - "MCP tools exist but aren't coordinated"
  - "Hard to describe 'build 10 landing pages' to AI"
needs:
  - MCP server that Claude Code can use
  - Natural language → site generation
  - Visible progress across parallel tasks
success: "Tell Claude 'build 10 landing pages' and it works"
```

---

## 5. Core Requirements

### Functional Requirements

#### FR1: Configuration Management

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR1.1 | Parse YAML configuration file | P0 | Valid YAML loads without errors |
| FR1.2 | Validate against JSON Schema | P0 | Invalid configs rejected with clear errors |
| FR1.3 | Support defaults with per-site overrides | P0 | Inheritance works correctly |
| FR1.4 | Environment variable substitution | P0 | `${VAR}` syntax works |
| FR1.5 | Config file watch for hot reload | P2 | Changes detected within 1s |

#### FR2: Site Generation

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR2.1 | Scaffold site from template | P0 | Files created in correct structure |
| FR2.2 | Apply brand tokens (colors, fonts) | P0 | Design system applied correctly |
| FR2.3 | Generate page structures from layout DSL | P0 | `hero+features+cta` expands correctly |
| FR2.4 | Parallel generation with concurrency control | P0 | Configurable 1-50 concurrent |
| FR2.5 | Incremental generation (skip unchanged) | P1 | Only modified sites regenerate |

#### FR3: CLI Interface

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR3.1 | `site.list` - List all configured sites | P0 | Outputs table/JSON of sites |
| FR3.2 | `site.plan` - Show what would be generated | P0 | Dry-run output matches execution |
| FR3.3 | `site.apply` - Execute generation steps | P0 | Sites generated correctly |
| FR3.4 | `site.view` - Show site at specific plane | P0 | All planes render correctly |
| FR3.5 | `site.sync-visual` - Sync with visual builder | P1 | Bidirectional sync works |
| FR3.6 | `site.deploy` - Trigger deployment | P1 | Deployment initiated |

#### FR4: MCP Integration

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR4.1 | MCP server exposing all CLI commands as tools | P0 | All tools callable |
| FR4.2 | Tool responses parseable by Claude | P0 | JSON/structured output |
| FR4.3 | Progress streaming for long operations | P1 | Real-time status updates |
| FR4.4 | Error handling with actionable messages | P0 | Errors include fix suggestions |

#### FR5: Visual Builder Bridge

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR5.1 | Pull content from Builder.io | P1 | Content appears in local files |
| FR5.2 | Push content to Builder.io | P1 | Changes visible in Builder |
| FR5.3 | Component registration bridge | P2 | Custom components available in Builder |
| FR5.4 | Conflict detection and resolution | P2 | Conflicts flagged before overwrite |

### Non-Functional Requirements

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| NFR1 | Generation speed | 100 sites in <10 min | Benchmark test |
| NFR2 | Memory usage | <2GB for 100 sites | Process monitoring |
| NFR3 | Error recovery | Resume after failure | Retry test |
| NFR4 | Cross-platform | macOS, Linux, Windows (WSL) | CI matrix |
| NFR5 | Node.js version | >=18.0.0 | Package.json engines |

---

## 6. Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   CLI        │    │  MCP Server  │    │  Future: Web Dashboard   │  │
│  │   (site.*)   │    │  (tools)     │    │  (React/Next.js)         │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────────────────┘  │
│         │                   │                                           │
│         └─────────┬─────────┘                                           │
│                   ↓                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                          COMMAND ROUTER                                 │
│                                                                         │
│  Parses commands → Validates input → Routes to handlers                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          CORE SERVICES                                  │
├───────────────┬───────────────┬───────────────┬─────────────────────────┤
│               │               │               │                         │
│  ┌────────────▼─────┐ ┌───────▼───────┐ ┌─────▼───────┐ ┌─────────────┐│
│  │ Config Manager   │ │ Orchestrator  │ │ Plane       │ │ Visual      ││
│  │                  │ │               │ │ Renderer    │ │ Bridge      ││
│  │ - Parse YAML     │ │ - Queue mgmt  │ │             │ │             ││
│  │ - Validate       │ │ - Parallel    │ │ - Business  │ │ - Builder   ││
│  │ - Resolve refs   │ │   execution   │ │ - Pages     │ │ - Plasmic   ││
│  │ - Watch changes  │ │ - Progress    │ │ - Components│ │ - Sync      ││
│  └────────────┬─────┘ │ - Retry       │ │ - Pipeline  │ │             ││
│               │       └───────┬───────┘ └─────────────┘ └─────────────┘│
│               │               │                                         │
├───────────────┴───────────────┴─────────────────────────────────────────┤
│                          GENERATOR ENGINE                               │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Template Engine │  │ Layout Expander │  │ Asset Processor │         │
│  │                 │  │                 │  │                 │         │
│  │ - Scaffolding   │  │ - hero+cta →    │  │ - Images        │         │
│  │ - File copy     │  │   components    │  │ - Fonts         │         │
│  │ - Token replace │  │ - Slot mapping  │  │ - Optimizing    │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          OUTPUT LAYER                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Turborepo Monorepo                           │   │
│  │                                                                 │   │
│  │  apps/                     packages/                            │   │
│  │  ├── alpha/               ├── ui/                              │   │
│  │  ├── beta/                ├── config/                          │   │
│  │  ├── gamma/               ├── tokens/                          │   │
│  │  └── ...100 sites         └── shared/                          │   │
│  │                                                                 │   │
│  │  turbo.json               package.json                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| **CLI** | Parse commands, format output | User commands | Structured calls |
| **MCP Server** | Expose tools for AI agents | MCP protocol | Tool responses |
| **Command Router** | Validate, route, log | Commands | Handler calls |
| **Config Manager** | Load, validate, watch config | YAML file | Typed config object |
| **Orchestrator** | Manage parallel execution | Task queue | Completion status |
| **Plane Renderer** | Generate abstraction views | Config + site data | Rendered view |
| **Visual Bridge** | Sync with Builder.io/Plasmic | Config | API calls |
| **Template Engine** | Scaffold sites from templates | Template + vars | Generated files |
| **Layout Expander** | Expand `hero+cta` notation | Layout string | Component tree |

---

## 7. Data Models

### Core Types

```typescript
// ============================================================
// SITE CONFIGURATION
// ============================================================

interface SiteConfig {
  id: string;                    // Unique identifier (kebab-case)
  name: string;                  // Human-readable name
  type: SiteType;                // marketing | docs | app | landing | ...
  audience?: string;             // Target audience description
  tone?: Tone;                   // professional | casual | playful | ...
  goals?: string[];              // Site goals
  language?: string;             // ISO language code (default: 'en')
  brand: BrandConfig;            // Brand/design tokens
  stack: StackConfig;            // Technology choices
  features: FeaturesConfig;      // Enabled features
  pages: Record<string, PageConfig>;  // Page definitions
  workflows: WorkflowConfig;     // Automation workflows
  metadata?: Record<string, unknown>; // Custom metadata
}

type SiteType =
  | 'marketing'
  | 'docs'
  | 'app'
  | 'landing'
  | 'blog'
  | 'ecommerce'
  | 'portfolio'
  | 'saas';

type Tone =
  | 'professional'
  | 'casual'
  | 'playful'
  | 'technical'
  | 'minimal';

// ============================================================
// BRAND CONFIGURATION
// ============================================================

interface BrandConfig {
  palette: PaletteName | 'custom';
  colors?: CustomColors;         // When palette='custom'
  font: FontName | 'custom';
  fontCustom?: string;           // When font='custom'
  logo?: string;                 // Path to logo
  favicon?: string;              // Path to favicon
}

type PaletteName =
  | 'ocean'
  | 'emerald'
  | 'sunset'
  | 'coral'
  | 'slate'
  | 'indigo'
  | 'rose'
  | 'amber';

interface CustomColors {
  primary: string;               // Hex color
  secondary: string;
  accent: string;
  background: string;
  foreground: string;
}

// ============================================================
// STACK CONFIGURATION
// ============================================================

interface StackConfig {
  framework: Framework;
  uiKit: UIKit;
  styling: StylingMethod;
  deploy: DeployTarget;
  docsEngine?: DocsEngine;
  auth?: AuthProvider;
  cms?: CMSProvider;
  database?: DatabaseProvider;
}

type Framework = 'next' | 'remix' | 'astro' | 'vite-spa' | 'nuxt' | 'sveltekit';
type UIKit = 'shadcn' | 'radix' | 'chakra' | 'mantine' | 'headless' | 'custom';
type StylingMethod = 'tailwind' | 'css-modules' | 'styled-components' | 'emotion';
type DeployTarget = 'vercel' | 'netlify' | 'cloudflare' | 'aws' | 'self-hosted';
type DocsEngine = 'contentlayer' | 'fumadocs' | 'nextra' | 'docusaurus' | 'astro-starlight';
type AuthProvider = 'clerk' | 'auth0' | 'nextauth' | 'supabase' | 'firebase' | 'none';
type CMSProvider = 'builder.io' | 'sanity' | 'contentful' | 'strapi' | 'payload' | 'mdx' | 'none';
type DatabaseProvider = 'postgres' | 'supabase' | 'planetscale' | 'turso' | 'mongodb' | 'none';

// ============================================================
// PAGE CONFIGURATION
// ============================================================

interface PageConfig {
  layout: string;                // Compressed layout: "hero+features+cta"
  source?: string;               // Content source path
  seedPosts?: number;            // AI-generated seed content count
  components?: ComponentOverride[];
}

interface ComponentOverride {
  slot: string;                  // Slot name to override
  component: string;             // Component path/name
  props?: Record<string, unknown>;
}

// ============================================================
// WORKFLOW CONFIGURATION
// ============================================================

// Workflows can be defined as:
// 1. Compact string with + notation: "scaffold+design+deploy"
// 2. Array of steps: ['scaffold', 'design', 'deploy']
// 3. Named reference: "init" (refers to workflows.init)

interface WorkflowConfig {
  // Named workflows (can be referenced by /mpg run <name>)
  [workflowName: string]: WorkflowDefinition;
}

type WorkflowDefinition =
  | string                           // Compact: "scaffold+design+deploy"
  | WorkflowStep[]                   // Array: ['scaffold', 'design']
  | PipelineDefinition;              // Complex pipeline with fork/wait

interface PipelineDefinition {
  on?: 'command' | 'push' | 'schedule';
  steps: PipelineStep[];
}

type PipelineStep =
  | WorkflowStep                     // Single step
  | { fork: WorkflowStep[] }         // Parallel execution
  | 'wait-all'                       // Barrier
  | { deploy: { env: string } };     // Parameterized step

type WorkflowStep =
  | 'scaffold'
  | 'design'
  | 'generate-copy'
  | 'generate-images'
  | 'wire-cms'
  | 'wire-docs'
  | 'stub-routes'
  | 'validate'
  | 'run-tests'
  | 'build'
  | 'deploy'
  | 'deploy-canary'
  | 'deploy-production'
  | 'sync-visual'
  | 'sync-cms'
  | 'regen-copy'
  | 'notify'
  | 'warm-cache'
  | 'invalidate-cdn'
  | 'run-lighthouse';

// Example usage in sites.yaml:
// workflows:
//   init: scaffold+design+generate-copy+deploy
//   update-copy: generate-copy
//   marketing-refresh:
//     on: command
//     steps:
//       - fork: [regen-copy, sync-visual]
//       - wait-all
//       - deploy:
//           env: production

// ============================================================
// ROOT CONFIGURATION
// ============================================================

interface MPGConfig {
  version: 'v1' | 'v2';
  defaults?: Partial<SiteConfig>;
  sites: SiteConfig[];
  views?: ViewsConfig;
  integrations?: IntegrationsConfig;
  orchestration?: OrchestrationConfig;
}

interface ViewsConfig {
  planes: Record<PlaneName, PlaneConfig>;
  defaultPlane?: PlaneName;
}

type PlaneName = 'business' | 'pages' | 'components' | 'pipeline' | 'deploy';

interface PlaneConfig {
  focus: string[];               // Fields to show
  source?: string;               // Source path for code views
  renderer: 'table' | 'tree' | 'graph' | 'code' | 'json' | 'kanban';
}

interface OrchestrationConfig {
  concurrency: number;           // 1-50
  batchSize: number;             // 1-100
  retryStrategy: 'exponential' | 'linear' | 'none';
  maxRetries: number;            // 0-5
}
```

### State Models

```typescript
// ============================================================
// GENERATION STATE
// ============================================================

interface GenerationJob {
  id: string;                    // Job UUID
  siteId: string;                // Site being generated
  status: JobStatus;
  steps: StepStatus[];
  startedAt: Date;
  completedAt?: Date;
  error?: GenerationError;
}

type JobStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

interface StepStatus {
  step: WorkflowStep;
  status: JobStatus;
  progress?: number;             // 0-100
  message?: string;
  startedAt?: Date;
  completedAt?: Date;
}

interface GenerationError {
  code: string;
  message: string;
  step?: WorkflowStep;
  recoverable: boolean;
  suggestion?: string;
}

// ============================================================
// ORCHESTRATION STATE
// ============================================================

interface OrchestratorState {
  jobs: Map<string, GenerationJob>;
  queue: string[];               // Job IDs waiting
  active: string[];              // Job IDs running
  completed: string[];           // Job IDs done
  failed: string[];              // Job IDs failed
  stats: OrchestratorStats;
}

interface OrchestratorStats {
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  averageDuration: number;       // ms
  estimatedRemaining: number;    // ms
}
```

---

## 8. DSL Specification

### YAML Configuration DSL

#### Minimal Example

```yaml
version: v1
sites:
  - id: my-site
    type: marketing
    pages:
      home:
        layout: hero+features+cta
```

#### Full Example with All Options

```yaml
version: v1

# Defaults inherited by all sites
defaults:
  stack:
    framework: next
    ui_kit: shadcn
    styling: tailwind
    deploy: vercel
  brand:
    palette: ocean
    font: inter
  features:
    blog: true
    dark_mode: true
    analytics: plausible

# Multi-plane view configuration
views:
  planes:
    business:
      focus: [id, name, type, goals, audience]
      renderer: table
    pages:
      focus: [id, pages.*, layout]
      renderer: tree
    components:
      source: packages/ui/src
      renderer: tree
    pipeline:
      source: turbo.json
      renderer: graph
  default_plane: business

# External integrations
integrations:
  builder:
    api_key: ${BUILDER_API_KEY}
    space_id: default-space
    sync:
      - site: alpha
        builder_entry: alpha-home
        page: home

# Parallel execution settings
orchestration:
  concurrency: 10
  batch_size: 20
  retry_strategy: exponential
  max_retries: 3

# Site definitions
sites:
  - id: alpha
    name: Alpha Analytics
    type: marketing
    audience: Data-driven startups
    goals: [lead-gen, newsletter]
    brand:
      palette: emerald
      logo: assets/alpha-logo.svg
    features:
      newsletter: true
      contact_form: true
    pages:
      home:
        layout: hero+features+social-proof+cta
      pricing:
        layout: pricing-3-tiers+faq+cta
      blog:
        layout: index+post
        seed_posts: 5
    workflows:
      on_init: [scaffold, design, generate-copy, deploy]
      on_change: [validate, build, deploy-canary]
```

### Layout DSL Specification

The `layout` field uses a **compressed notation** where components are joined by `+`:

```
layout: hero+features+social-proof+cta
         │      │           │        │
         │      │           │        └── Call-to-action section
         │      │           └── Testimonials/logos
         │      └── Feature grid/list
         └── Hero section with headline
```

#### Available Blocks

| Block | Description | Variants |
|-------|-------------|----------|
| `hero` | Main hero section | `hero-video`, `hero-minimal`, `hero-image` |
| `features` | Feature showcase | `features-grid`, `features-list`, `features-cards` |
| `cta` | Call to action | `cta-centered`, `cta-split`, `cta-banner` |
| `pricing` | Pricing tables | `pricing-3-tiers`, `pricing-comparison` |
| `faq` | FAQ section | `faq-accordion`, `faq-grid` |
| `testimonials` | Customer quotes | `testimonials-carousel`, `testimonials-grid` |
| `social-proof` | Logos, stats | `logos`, `stats`, `badges` |
| `team` | Team members | `team-grid`, `team-list` |
| `blog` | Blog layout | `index`, `post`, `featured` |
| `docs` | Documentation | `sidebar+content`, `sidebar+content+toc` |
| `form` | Contact/signup | `form`, `form+calendar` |

---

## 9. CLI Command Grammar

### Design Philosophy

The command grammar follows a **minimal, machine-precise** pattern:

```
verb + target + modifiers
```

This is inspired by GitHub slash commands and workflow DSLs—low keystroke, high clarity.

### Primary Syntax

```bash
/mpg <verb> <target> [modifiers...]
```

Or with the short alias:

```bash
/m <verb> <target> [modifiers...]
```

### Core Verbs

| Verb | Description | Example |
|------|-------------|---------|
| `plan` | Preview what would be generated (dry run) | `/mpg plan sites type=marketing` |
| `apply` | Execute generation steps | `/mpg apply sites ids=alpha,beta steps=scaffold+design` |
| `deploy` | Trigger deployment | `/mpg deploy sites type=docs env=staging` |
| `sync` | Sync with visual builder | `/mpg sync site=alpha from=builder` |
| `view` | Show site at abstraction plane | `/mpg view site=alpha plane=pages` |
| `list` | List configured sites | `/mpg list sites type=marketing` |
| `run` | Run named workflow | `/mpg run init type=marketing` |
| `status` | Show generation status | `/mpg status` |
| `diff` | Compare versions/planes | `/mpg diff alpha@current alpha@builder` |

### Targets

| Target | Syntax | Description |
|--------|--------|-------------|
| All sites | `sites` | All configured sites |
| By type | `sites type=marketing` | Filter by site type |
| Single site | `site=alpha` | Specific site by ID |
| Multiple sites | `ids=alpha,beta,gamma` | Comma-separated IDs |
| By goal | `sites goal=lead-gen` | Filter by goal |
| Limit | `sites limit=20` | First N sites |

### Modifiers

| Modifier | Description | Example |
|----------|-------------|---------|
| `steps=` | Workflow steps (use `+` for multiple) | `steps=scaffold+design+deploy` |
| `env=` | Deployment environment | `env=staging` |
| `plane=` | View abstraction plane | `plane=business` |
| `from=` / `to=` | Sync direction | `from=builder` |
| `format=` | Output format | `format=json` |
| `--dry` | Preview without executing | `--dry` |
| `--verbose` | Detailed output | `--verbose` |

### Compact Workflow Blocks (YAML)

Inside `sites.yaml`, workflows use the `+` notation for brevity:

```yaml
workflows:
  init: scaffold+design+generate-copy+deploy
  update-copy: generate-copy
  full-redeploy: test+build+deploy
  visual-sync: sync-visual+validate+deploy
```

Then execute with:

```bash
/mpg run init type=marketing
/mpg run update-copy ids=alpha,gamma
/mpg run full-redeploy --all
```

### Pipeline Syntax (Advanced Flows)

For complex orchestration with parallelism:

```yaml
pipelines:
  marketing-refresh:
    on: command
    steps:
      - fork: [regen-copy, sync-visual]   # Run in parallel
      - wait-all                           # Barrier
      - deploy env=production              # Sequential

  docs-publish:
    steps:
      - validate
      - build
      - deploy env=docs
```

Execute with:

```bash
/mpg run marketing-refresh type=marketing
/mpg run docs-publish ids=beta-docs
```

### Sentence Mode (Natural Language)

For conversational/MCP interfaces, free-form sentences compile to verbs:

```bash
/mpg "scaffold 20 marketing sites with ocean theme"
/mpg "sync visual changes from builder and redeploy docs"
/mpg "promote all staging sites to production"
/mpg "refresh copy for all landing pages"
```

**Parser mapping:**

| Sentence Pattern | Compiled Command |
|-----------------|------------------|
| "scaffold 20 marketing sites" | `apply sites type=marketing limit=20 steps=scaffold` |
| "sync from builder" | `sync from=builder` |
| "deploy to production" | `deploy env=production` |
| "refresh copy for landing" | `run update-copy type=landing` |

### Command Examples (Complete)

```bash
# List all marketing sites
/mpg list sites type=marketing

# Preview generating 10 sites
/mpg plan sites limit=10 --dry

# Scaffold and design specific sites
/mpg apply ids=alpha,beta steps=scaffold+design

# Deploy all docs sites to staging
/mpg deploy sites type=docs env=staging

# View alpha at the pages plane
/mpg view site=alpha plane=pages

# Sync visual builder content
/mpg sync site=alpha from=builder

# Run the full init workflow
/mpg run init type=marketing

# Check status of all jobs
/mpg status

# Compare current state vs builder state
/mpg diff alpha@current alpha@builder
```

### Grammar (EBNF)

```ebnf
command     = "/mpg" verb target? modifier* ;
verb        = "plan" | "apply" | "deploy" | "sync" | "view"
            | "list" | "run" | "status" | "diff" ;
target      = "sites" | "site=" id | "ids=" id-list ;
id-list     = id ("," id)* ;
modifier    = key "=" value | "--" flag ;
key         = "type" | "steps" | "env" | "plane" | "from" | "to"
            | "limit" | "format" | "goal" ;
value       = word | step-chain ;
step-chain  = step ("+" step)* ;
step        = "scaffold" | "design" | "generate-copy" | "deploy" | ... ;
flag        = "dry" | "verbose" | "quiet" | "all" ;
id          = [a-z][a-z0-9-]* ;
```

---

## 10. MCP Tool Interfaces

### Tool Definitions

Each CLI command maps to an MCP tool:

```typescript
// ============================================================
// MCP TOOL: list_sites
// ============================================================

interface ListSitesTool {
  name: 'list_sites';
  description: 'List all configured sites with optional filtering';
  inputSchema: {
    type: 'object';
    properties: {
      type?: SiteType;
      goal?: string;
      format?: 'table' | 'json' | 'ids';
    };
  };
  outputSchema: {
    type: 'array';
    items: SiteSummary;
  };
}

interface SiteSummary {
  id: string;
  name: string;
  type: SiteType;
  goals: string[];
  status: 'configured' | 'generated' | 'deployed';
}

// ============================================================
// MCP TOOL: plan_sites
// ============================================================

interface PlanSitesTool {
  name: 'plan_sites';
  description: 'Preview what would be generated (dry run)';
  inputSchema: {
    type: 'object';
    properties: {
      ids?: string[];
      type?: SiteType;
      steps?: WorkflowStep[];
    };
  };
  outputSchema: {
    type: 'object';
    properties: {
      sites: PlanResult[];
      summary: PlanSummary;
    };
  };
}

interface PlanResult {
  siteId: string;
  steps: PlannedStep[];
  estimatedDuration: number;
  filesAffected: string[];
}

// ============================================================
// MCP TOOL: apply_sites
// ============================================================

interface ApplySitesTool {
  name: 'apply_sites';
  description: 'Execute generation steps for selected sites';
  inputSchema: {
    type: 'object';
    properties: {
      ids?: string[];
      type?: SiteType;
      steps: WorkflowStep[];
      dryRun?: boolean;
    };
    required: ['steps'];
  };
  outputSchema: {
    type: 'object';
    properties: {
      jobId: string;
      status: 'started' | 'completed' | 'failed';
      results: ApplyResult[];
    };
  };
}

// ============================================================
// MCP TOOL: get_site_plane
// ============================================================

interface GetSitePlaneTool {
  name: 'get_site_plane';
  description: 'Get site view at specific abstraction plane';
  inputSchema: {
    type: 'object';
    properties: {
      id: string;
      plane: PlaneName;
    };
    required: ['id', 'plane'];
  };
  outputSchema: {
    type: 'object';
    properties: {
      plane: PlaneName;
      content: PlaneContent;
      format: 'table' | 'tree' | 'graph' | 'json';
    };
  };
}

// ============================================================
// MCP TOOL: sync_visual
// ============================================================

interface SyncVisualTool {
  name: 'sync_visual';
  description: 'Sync site with visual builder (Builder.io/Plasmic)';
  inputSchema: {
    type: 'object';
    properties: {
      id: string;
      direction: 'to-builder' | 'from-builder' | 'bidirectional';
      pages?: string[];
    };
    required: ['id', 'direction'];
  };
  outputSchema: {
    type: 'object';
    properties: {
      synced: SyncResult[];
      conflicts: ConflictResult[];
    };
  };
}

// ============================================================
// MCP TOOL: get_generation_status
// ============================================================

interface GetGenerationStatusTool {
  name: 'get_generation_status';
  description: 'Get current status of site generation jobs';
  inputSchema: {
    type: 'object';
    properties: {
      jobId?: string;
    };
  };
  outputSchema: {
    type: 'object';
    properties: {
      jobs: GenerationJob[];
      stats: OrchestratorStats;
    };
  };
}
```

### MCP Server Configuration

```json
{
  "mcpServers": {
    "multi-project-generator": {
      "command": "npx",
      "args": ["@meta-prompting/mpg", "mcp:start"],
      "env": {
        "MPG_CONFIG": "./sites.yaml",
        "BUILDER_API_KEY": "${BUILDER_API_KEY}"
      }
    }
  }
}
```

---

## 11. Multi-Plane View System

### Concept

The **multi-plane view system** allows viewing the same project at different abstraction levels:

```
┌─────────────────────────────────────────────────────────────┐
│                     ABSTRACTION LEVELS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PLANE 1: BUSINESS                                    │   │
│  │                                                      │   │
│  │  "What are we building and why?"                     │   │
│  │                                                      │   │
│  │  • Site names, types, goals                          │   │
│  │  • Target audiences                                  │   │
│  │  • Business metrics                                  │   │
│  │                                                      │   │
│  │  Renderer: TABLE                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PLANE 2: PAGES                                       │   │
│  │                                                      │   │
│  │  "What pages exist and how are they structured?"     │   │
│  │                                                      │   │
│  │  • Page tree for each site                           │   │
│  │  • Layout compositions                               │   │
│  │  • Content sources                                   │   │
│  │                                                      │   │
│  │  Renderer: TREE                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PLANE 3: COMPONENTS                                  │   │
│  │                                                      │   │
│  │  "What components make up each page?"                │   │
│  │                                                      │   │
│  │  • Component hierarchy                               │   │
│  │  • Props and variants                                │   │
│  │  • Shared vs. site-specific                          │   │
│  │                                                      │   │
│  │  Renderer: TREE / CODE                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PLANE 4: PIPELINE                                    │   │
│  │                                                      │   │
│  │  "How is this built and deployed?"                   │   │
│  │                                                      │   │
│  │  • Turborepo task graph                              │   │
│  │  • Build dependencies                                │   │
│  │  • Deploy targets                                    │   │
│  │                                                      │   │
│  │  Renderer: GRAPH                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Plane Definitions

| Plane | Purpose | Fields | Renderer |
|-------|---------|--------|----------|
| **business** | Strategic view | id, name, type, goals, audience | Table |
| **pages** | Content architecture | pages.*, layout, source | Tree |
| **components** | UI composition | component hierarchy, props | Tree/Code |
| **pipeline** | Build/deploy | turbo tasks, dependencies | Graph |
| **deploy** | Operations | deploy status, environments | Kanban |

### Example Outputs

#### Business Plane (Table)

```
┌──────────────┬───────────────────┬───────────┬─────────────────────┐
│ ID           │ Name              │ Type      │ Goals               │
├──────────────┼───────────────────┼───────────┼─────────────────────┤
│ alpha        │ Alpha Analytics   │ marketing │ lead-gen, newsletter│
│ beta-docs    │ Beta SDK Docs     │ docs      │ education           │
│ gamma-app    │ Gamma Dashboard   │ app       │ retention           │
└──────────────┴───────────────────┴───────────┴─────────────────────┘
```

#### Pages Plane (Tree)

```
alpha/
├── home
│   └── layout: hero + features + social-proof + cta
├── pricing
│   └── layout: pricing-3-tiers + faq + cta
├── about
│   └── layout: team + values + timeline
└── blog/
    ├── index
    └── [slug]
        └── seed_posts: 5
```

#### Pipeline Plane (Graph)

```
                    ┌─────────────┐
                    │   lint      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │ build:ui  │ │build:cfg│ │build:tokens│
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │ build:sites │
                    │  (parallel) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   deploy    │
                    └─────────────┘
```

---

## 12. Visual Builder Integration

### Builder.io Integration

#### Sync Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BUILDER.IO SYNC FLOW                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LOCAL (sites.yaml + code)              BUILDER.IO                   │
│                                                                      │
│  ┌─────────────────────┐               ┌─────────────────────┐      │
│  │ sites.yaml          │               │ Builder Space       │      │
│  │ └─ pages:           │               │ └─ Models:          │      │
│  │     └─ home:        │  ──────────▶  │     └─ page         │      │
│  │         layout: ... │  (to-builder) │         └─ entries  │      │
│  └─────────────────────┘               └─────────────────────┘      │
│                                                                      │
│  ┌─────────────────────┐               ┌─────────────────────┐      │
│  │ apps/alpha/         │               │ Builder Editor      │      │
│  │ └─ pages/           │  ◀──────────  │ └─ Visual blocks    │      │
│  │     └─ home.tsx     │ (from-builder)│     └─ Content      │      │
│  └─────────────────────┘               └─────────────────────┘      │
│                                                                      │
│  SYNC OPERATIONS:                                                    │
│  • to-builder: Push layout blocks as Builder entries                │
│  • from-builder: Pull Builder content as local components           │
│  • bidirectional: Two-way sync with conflict detection              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### Configuration

```yaml
integrations:
  builder:
    api_key: ${BUILDER_API_KEY}
    space_id: default-space
    model: page
    sync:
      - site: alpha
        builder_entry: alpha-home
        page: home
        direction: bidirectional
```

#### API Mapping

| MPG Action | Builder.io API |
|------------|----------------|
| Push layout | `POST /api/v1/write/{model}` |
| Pull content | `GET /api/v1/content/{model}` |
| List entries | `GET /api/v1/content/{model}?limit=100` |
| Update entry | `PATCH /api/v1/write/{model}/{id}` |

### Conflict Resolution

When bidirectional sync detects changes on both sides:

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFLICT DETECTED                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local change:   hero.headline = "New Headline"             │
│  Builder change: hero.headline = "Different Headline"       │
│                                                             │
│  Options:                                                   │
│  [1] Keep local (overwrite Builder)                         │
│  [2] Keep Builder (overwrite local)                         │
│  [3] Merge (create hero.headline_local, hero.headline_builder)│
│  [4] Skip (leave conflict unresolved)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Orchestration Engine

### Parallel Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION ENGINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: 100 sites to generate                                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      JOB QUEUE                               │   │
│  │                                                              │   │
│  │  [site-001] [site-002] [site-003] ... [site-100]            │   │
│  │                                                              │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CONCURRENCY CONTROLLER                          │   │
│  │              (max: 10 concurrent)                            │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐                │
│         │         │          │          │         │                │
│         ▼         ▼          ▼          ▼         ▼                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ...          │
│  │ Worker 1 │ │ Worker 2 │ │ Worker 3 │ │ Worker 4 │              │
│  │ site-001 │ │ site-002 │ │ site-003 │ │ site-004 │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       │            │            │            │                     │
│       ▼            ▼            ▼            ▼                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │                    STEP EXECUTOR                          │     │
│  │                                                           │     │
│  │  scaffold → design → generate-copy → deploy              │     │
│  │                                                           │     │
│  └──────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │                    PROGRESS TRACKER                       │     │
│  │                                                           │     │
│  │  [████████████████████░░░░░░░░░░] 65/100 sites (12m 34s) │     │
│  │                                                           │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Retry Strategy

```typescript
interface RetryConfig {
  strategy: 'exponential' | 'linear' | 'none';
  maxRetries: number;           // 0-5
  baseDelayMs: number;          // 1000
  maxDelayMs: number;           // 30000
}

// Exponential backoff example:
// Attempt 1: immediate
// Attempt 2: 1s delay
// Attempt 3: 2s delay
// Attempt 4: 4s delay
// Attempt 5: 8s delay (capped at maxDelayMs)
```

### Failure Handling

| Failure Type | Behavior |
|--------------|----------|
| Transient (network) | Retry with backoff |
| Permanent (validation) | Skip site, continue others |
| Catastrophic (disk full) | Halt all, report |

---

## 14. Template System

### Template Structure

```
templates/
├── next-marketing/
│   ├── template.yaml           # Template metadata
│   ├── base/                   # Base files (always copied)
│   │   ├── package.json
│   │   ├── next.config.js
│   │   ├── tsconfig.json
│   │   └── tailwind.config.js
│   ├── components/             # Component library
│   │   ├── hero/
│   │   ├── features/
│   │   ├── cta/
│   │   └── ...
│   ├── layouts/                # Page layouts
│   │   ├── marketing.tsx
│   │   └── blog.tsx
│   └── styles/                 # Base styles
│       └── globals.css
├── next-docs/
├── next-app/
├── remix-app/
└── astro-landing/
```

### Template Metadata

```yaml
# template.yaml
name: next-marketing
version: 1.0.0
description: Next.js marketing site template
framework: next
compatible_types: [marketing, landing]
variables:
  - name: siteName
    type: string
    required: true
  - name: primaryColor
    type: color
    default: "#3B82F6"
dependencies:
  - next@14
  - react@18
  - tailwindcss@3
blocks:
  - hero
  - hero-video
  - features-grid
  - features-list
  - pricing-3-tiers
  - testimonials-carousel
  - cta-centered
  - footer
```

### Variable Substitution

Templates use `{{variable}}` syntax:

```tsx
// components/hero.tsx (template)
export function Hero() {
  return (
    <section className="bg-{{brand.palette}}-500">
      <h1>{{site.name}}</h1>
      <p>{{site.tagline}}</p>
    </section>
  );
}

// After generation (apps/alpha/components/hero.tsx)
export function Hero() {
  return (
    <section className="bg-emerald-500">
      <h1>Alpha Analytics</h1>
      <p>Data-driven insights for growth teams</p>
    </section>
  );
}
```

---

## 15. Security Considerations

### API Key Management

| Concern | Mitigation |
|---------|------------|
| Keys in config | Use `${ENV_VAR}` substitution |
| Keys in logs | Redact in all output |
| Keys in git | `.gitignore` and pre-commit hooks |

### Input Validation

| Input | Validation |
|-------|------------|
| Site IDs | `/^[a-z][a-z0-9-]*$/` |
| File paths | No `..` traversal |
| URLs | HTTPS only, allowlist domains |
| Shell commands | No shell injection possible (no exec) |

### Rate Limiting

| Service | Limit | Handling |
|---------|-------|----------|
| Builder.io API | 100 req/min | Queue with delays |
| GitHub API | 5000 req/hr | Token rotation |
| Deploy providers | Varies | Configurable throttle |

---

## 16. Open Questions

These require your input before proceeding:

### Q1: Monorepo vs. Polyrepo Output

**Options:**
- **A) Turborepo monorepo** (all 100 sites in one repo)
  - Pros: Shared deps, atomic deploys, unified CI
  - Cons: Large repo, permission complexity

- **B) Polyrepo** (each site gets its own repo)
  - Pros: Independent versioning, simpler permissions
  - Cons: Dependency drift, manual coordination

**Your preference?** _______________

### Q2: Default Framework

**Options:**
- A) Next.js 14 (App Router)
- B) Next.js 14 (Pages Router)
- C) Remix
- D) Astro
- E) Vite SPA

**Your preference?** _______________

### Q3: Default UI Kit

**Options:**
- A) shadcn/ui (copy-paste components)
- B) Radix Primitives (headless)
- C) Chakra UI
- D) Custom/none

**Your preference?** _______________

### Q4: AI Content Generation

Should MPG include AI content generation (headlines, copy, images)?

**Options:**
- A) Yes, built-in (Claude API calls)
- B) Yes, via MCP tool delegation
- C) No, content is external
- D) Optional plugin

**Your preference?** _______________

### Q5: Visual Builder Priority

Which visual builder to prioritize first?

**Options:**
- A) Builder.io (most feature-rich)
- B) Plasmic (React-native)
- C) Both equally
- D) None initially (CLI-only v1)

**Your preference?** _______________

### Q6: Deployment Integration

**Options:**
- A) Vercel-first (with others later)
- B) Netlify-first
- C) Provider-agnostic from start
- D) No deployment in v1 (generate only)

**Your preference?** _______________

---

## 17. Approval Checklist

Please review and approve each section:

| Section | Status | Notes |
|---------|--------|-------|
| 1. Executive Summary | ⬜ Pending | |
| 2. Problem Statement | ⬜ Pending | |
| 3. Goals & Non-Goals | ⬜ Pending | |
| 4. User Personas | ⬜ Pending | |
| 5. Core Requirements | ⬜ Pending | |
| 6. Architecture Overview | ⬜ Pending | |
| 7. Data Models | ⬜ Pending | |
| 8. DSL Specification | ⬜ Pending | |
| 9. CLI Command Grammar | ⬜ Pending | |
| 10. MCP Tool Interfaces | ⬜ Pending | |
| 11. Multi-Plane View System | ⬜ Pending | |
| 12. Visual Builder Integration | ⬜ Pending | |
| 13. Orchestration Engine | ⬜ Pending | |
| 14. Template System | ⬜ Pending | |
| 15. Security Considerations | ⬜ Pending | |
| Open Questions Answered | ⬜ Pending | |

---

## Next Steps (After Approval)

1. **Finalize open questions** with your input
2. **Lock specification** (version 0.1.0)
3. **Create implementation tickets** from requirements
4. **Begin Phase 1 implementation**:
   - Core types and schema parser
   - CLI skeleton with first 3 commands
   - Basic MCP server shell

---

**Document Status:** AWAITING APPROVAL

Please review this specification and provide:
1. Answers to the 6 open questions
2. Approval/rejection for each section
3. Any modifications or additions needed

I will not write any implementation code until this specification is approved.
