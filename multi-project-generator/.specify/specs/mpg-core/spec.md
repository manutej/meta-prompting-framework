# Feature Specification: MPG Core System

**Version:** 0.2.0-draft
**Status:** AWAITING THREE-PLANE APPROVAL
**Last Updated:** 2026-01-04

---

## Executive Summary

**One Sentence**: Enable agencies to generate 5-10 similar websites from one YAML configuration, reducing scaffolding time by 80%.

**MERCURIO Wisdom Check**:
- **Mental**: Solves documented problem (boilerplate repetition)
- **Physical**: Achievable in 8-10 weeks
- **Spiritual**: Serves agency developers directly

---

## 1. MENTAL PLANE: Understanding What We're Building

### 1.1 Problem Analysis

**Validated Problem**: Developers building multiple similar sites spend excessive time on repetitive scaffolding and configuration.

**Evidence**:
- [ASSUMPTION] Agency developers create 3-5 similar sites per month
- [ASSUMPTION] Each site takes 2-3 days to scaffold and configure
- [VALIDATED] No existing tool combines: YAML config + parallel generation + MCP integration
- [VALIDATED] Turborepo proves monorepo-at-scale is feasible

**Pattern Recognition**:
- Similar to Terraform (declarative → infrastructure)
- Similar to Docker Compose (one config → multiple services)
- Novel: AI-native via MCP integration

### 1.2 Assumptions Registry

| ID | Assumption | Validation Method | Status |
|----|------------|-------------------|--------|
| A1 | Agencies need 5-10 sites/month | User interviews | UNVALIDATED |
| A2 | YAML is preferred over GUI | User interviews | UNVALIDATED |
| A3 | 100 sites in <10 min is achievable | Benchmark test | UNVALIDATED |
| A4 | MCP integration adds value | Claude Code usage data | PARTIAL |
| A5 | Visual builder sync is desired | User interviews | UNVALIDATED |

**Risk**: Proceeding with unvalidated assumptions. Mitigation: Interview 3-5 target users before Phase 1.

### 1.3 Success Criteria (Measurable)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time savings | 80% reduction in scaffolding time | Before/after comparison |
| Configuration clarity | Config readable by non-author in 5 min | User test |
| Parallel efficiency | 10 sites generate in <3 min | Benchmark |
| Error recovery | Failed sites don't block others | Integration test |
| MCP completeness | 100% CLI parity | Feature matrix |

---

## 2. PHYSICAL PLANE: Reality & Constraints

### 2.1 Resource Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| 1 developer | Limited parallelization | Strict MVP scope |
| 8-10 week timeline | No advanced features | Phase 1+ deferral |
| No dedicated QA | Developer-driven testing | TDD approach |
| Open source | No commercial integrations | MIT license only |

### 2.2 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Concurrent execution complexity | High | High | Use proven queue library (p-limit) |
| Builder.io API changes | Medium | Medium | Defer to Phase 2 |
| Turborepo monorepo performance | Medium | High | Test with 50 sites first |
| MCP protocol instability | Low | Medium | Minimal tool surface |

### 2.3 MVP Scope (Phase 0)

**IN SCOPE** (must ship):
- YAML configuration parsing + validation
- Parallel scaffolding (configurable 1-20 concurrent)
- 5 CLI commands: `list`, `plan`, `apply`, `status`, `view`
- 3 MCP tools: `list_sites`, `plan_sites`, `apply_sites`
- 2 templates: `next-marketing`, `next-docs`

**OUT OF SCOPE** (Phase 1+):
- Visual builder integration (Builder.io, Plasmic)
- Multi-plane view system (4 abstraction layers)
- Advanced workflow orchestration (fork/wait-all)
- AI content generation
- Deployment integration (Vercel, Netlify)

### 2.4 Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 0 (MVP) | 8-10 weeks | Core CLI + MCP + 2 templates |
| Phase 1 | 4-6 weeks | Deployment + more templates |
| Phase 2 | 6-8 weeks | Visual builder + advanced workflows |

---

## 3. SPIRITUAL PLANE: Purpose & Values

### 3.1 Who Are We Serving?

**PRIMARY Persona (v1 Target)**: Alex - Agency Developer

```yaml
name: Alex
role: Full-stack developer at digital agency
problem: "I spend 2-3 days scaffolding each of 5 monthly projects"
desired_outcome: "Generate 5 sites in an afternoon"
success_measure: "Time to first deploy < 4 hours"
```

**SECONDARY Persona (v2 Target)**: Jordan - Platform Engineer

```yaml
name: Jordan
role: Platform engineer at SaaS company
problem: "50+ tenant marketing pages need coordinated updates"
desired_outcome: "Update branding across all sites in one command"
success_measure: "Batch update < 30 minutes"
```

**DEFERRED Persona (v3+)**: Sam - AI-Augmented Builder

```yaml
name: Sam
role: Indie hacker using Claude Code
problem: "I want Claude to build my sites but need structure"
desired_outcome: "Tell Claude 'build 10 landing pages' and it works"
success_measure: "Natural language → working sites"
```

### 3.2 Values Alignment

| Value | How MPG Honors It |
|-------|-------------------|
| **Accessibility** | CLI-first (no GUI required), simple YAML |
| **Transparency** | Open source, clear progress output |
| **Autonomy** | No vendor lock-in (Article V) |
| **Efficiency** | Parallel execution, reusable templates |
| **Trust** | Security by design (Article VI) |

### 3.3 Unintended Consequences Analysis

| Consequence | Type | Mitigation |
|-------------|------|------------|
| Framework becomes barrier to entry | Negative | Keep config simple (< 50 lines for basic site) |
| Users locked into MPG patterns | Negative | Export as standard Next.js/Remix project |
| Visual builder dependency | Negative | Make visual sync optional (Phase 2) |
| Enables rapid prototyping | Positive | Feature, not bug |
| Reduces agency revenue per site | Neutral | Value shift from scaffolding to customization |

### 3.4 Legacy Impact

**5-Year Vision**: MPG is the standard for generating multiple sites, like Terraform is for infrastructure.

**Sustainability Check**:
- [ ] Can future maintainers understand the codebase?
- [ ] Are decisions documented with rationale (ADRs)?
- [ ] Is the architecture extensible without rewrites?

---

## 4. THREE-PLANE CONVERGENCE

### 4.1 Where All Planes Agree (High Confidence)

| Feature | Mental | Physical | Spiritual |
|---------|--------|----------|-----------|
| YAML config for multiple sites | Proven pattern | Simple to implement | Serves user need |
| Parallel scaffolding | Technically sound | p-limit library exists | Time savings |
| CLI commands (list/plan/apply) | Clear semantics | Well-scoped | Developer-friendly |
| MCP integration | AI-native approach | MCP SDK stable | Enables agentic workflows |
| Template system | Reusability pattern | Standard tooling | Reduces repetition |

**Recommendation**: Build these first.

### 4.2 Where Planes Conflict (Risk Zones)

#### Conflict 1: 100 Sites vs. 10 Sites

| Plane | Position |
|-------|----------|
| Mental | "100 sites is impressive scale" |
| Physical | "100 sites has different failure modes than 10" |
| Spiritual | "Alex only needs 5-10; who needs 100?" |

**Resolution**: Design for 10, test with 50, market as 100. MVP targets 10-20 concurrent.

#### Conflict 2: Multi-Plane Views

| Plane | Position |
|-------|----------|
| Mental | "4 abstraction layers is elegant" |
| Physical | "4 renderers = 4 weeks of work" |
| Spiritual | "No user has asked for this" |

**Resolution**: Defer to Phase 2. MVP has one view: `list` output.

#### Conflict 3: Visual Builder Sync

| Plane | Position |
|-------|----------|
| Mental | "Bidirectional sync is powerful" |
| Physical | "Builder.io API + conflict resolution = 6 weeks" |
| Spiritual | "Locks users into Builder.io ecosystem" |

**Resolution**: Defer to Phase 2. Gather user demand data first.

### 4.3 Wisdom Synthesis

> "Ship a focused MVP that solves Alex's core problem: generate 5-10 sites from one config. Prove value quickly. Expand based on real usage data, not imagined features."

---

## 5. User Stories (P1/P2/P3)

### P1 (Must Have) - Generate Multiple Sites from Config

**User**: Alex (Agency Developer)

**Journey**:
1. Alex creates `sites.yaml` with 5 marketing site configurations
2. Alex runs `/mpg plan sites` to preview what will be generated
3. Alex runs `/mpg apply sites steps=scaffold+design`
4. System generates 5 site directories in parallel
5. Each site is independently buildable with `npm run build`

**Success Criteria**: All 5 sites scaffold correctly in <2 minutes

**Acceptance Tests**:
```gherkin
Given a valid sites.yaml with 5 marketing sites
When /mpg apply sites steps=scaffold runs
Then 5 directories are created in apps/
And each directory has valid package.json
And npm install succeeds for each site
And total execution time is <2 minutes
```

**Unknowns**:
- [NEEDS CLARIFICATION] Should sites share node_modules (Turborepo) or be independent?

---

### P1 (Must Have) - Preview Before Generation

**User**: Alex (Agency Developer)

**Journey**:
1. Alex runs `/mpg plan sites type=marketing`
2. System shows what would be generated (dry run)
3. Alex reviews and adjusts config if needed
4. Alex runs `/mpg apply` only when satisfied

**Success Criteria**: Plan output matches actual generation

**Acceptance Tests**:
```gherkin
Given a valid sites.yaml
When /mpg plan sites --dry runs
Then output shows list of sites to generate
And output shows steps that will execute
And no files are created on disk
```

---

### P1 (Must Have) - MCP Integration for Claude Code

**User**: Sam (AI-Augmented Builder)

**Journey**:
1. Sam configures MPG as MCP server in Claude Code
2. Sam asks Claude: "Generate 3 landing pages for AI tools"
3. Claude calls `list_sites`, `plan_sites`, `apply_sites` tools
4. Sites are generated via Claude's orchestration

**Success Criteria**: All CLI commands available as MCP tools

**Acceptance Tests**:
```gherkin
Given MPG is configured as MCP server
When Claude calls list_sites tool
Then response contains array of site summaries
And response is valid JSON
```

---

### P2 (Should Have) - View Site Status

**User**: Alex (Agency Developer)

**Journey**:
1. Alex runs `/mpg status` during generation
2. System shows progress for each site
3. Alex can see which sites completed, which failed

**Success Criteria**: Real-time status visibility

**Acceptance Tests**:
```gherkin
Given generation is in progress
When /mpg status runs
Then output shows per-site status (pending/running/completed/failed)
And output shows overall progress percentage
```

---

### P2 (Should Have) - Custom Templates

**User**: Jordan (Platform Engineer)

**Journey**:
1. Jordan creates custom template in `templates/` directory
2. Jordan references template in sites.yaml: `template: custom-saas`
3. Sites generate using Jordan's custom template

**Success Criteria**: Custom templates work like built-in templates

---

### P3 (Nice to Have) - Named Workflow Execution

**User**: Alex (Agency Developer)

**Journey**:
1. Alex defines named workflow: `init: scaffold+design+deploy`
2. Alex runs `/mpg run init type=marketing`
3. System executes all steps in workflow

**Success Criteria**: Workflows are reusable across sites

---

### P3 (Nice to Have) - Incremental Generation

**User**: Jordan (Platform Engineer)

**Journey**:
1. Jordan modifies 2 of 50 site configs
2. Jordan runs `/mpg apply --incremental`
3. Only the 2 modified sites regenerate

**Success Criteria**: Skip unchanged sites

---

## 6. Functional Requirements

### FR-001: Configuration Parsing

The system MUST parse YAML configuration files following the sites schema.

**Acceptance Criteria**:
- Valid YAML loads without errors
- Invalid YAML returns clear error with line number
- Environment variables (${VAR}) are substituted
- Defaults are inherited by sites

**Affected Stories**: P1 (Generate), P1 (Preview)

---

### FR-002: Schema Validation

The system MUST validate configurations against JSON Schema before execution.

**Acceptance Criteria**:
- Schema violations return specific field errors
- Required fields are enforced
- Type mismatches are caught
- Unknown fields generate warnings (not errors)

**Affected Stories**: P1 (Generate), P1 (Preview)

---

### FR-003: Parallel Execution

The system MUST support parallel site generation with configurable concurrency.

**Acceptance Criteria**:
- Concurrency is configurable (1-50)
- Default concurrency is 10
- Failed sites do not block others
- Progress is trackable per-site

**Affected Stories**: P1 (Generate), P2 (Status)

---

### FR-004: Template Scaffolding

The system MUST scaffold sites from templates with variable substitution.

**Acceptance Criteria**:
- Templates are copied to output directory
- Variables ({{site.name}}, {{brand.palette}}) are replaced
- Brand tokens apply to design system files
- Generated sites are independently buildable

**Affected Stories**: P1 (Generate), P2 (Custom Templates)

---

### FR-005: CLI Interface

The system MUST provide CLI commands following the `/mpg verb target modifiers` grammar.

**Acceptance Criteria**:
- 5 commands: list, plan, apply, status, view
- Commands support selectors: type=, ids=, site=
- Output formats: table (default), json
- Help text for each command

**Affected Stories**: All

---

### FR-006: MCP Tool Exposure

The system MUST expose CLI functionality as MCP tools.

**Acceptance Criteria**:
- 3 tools minimum: list_sites, plan_sites, apply_sites
- Tool responses are valid JSON
- Errors include actionable messages
- Tools are discoverable via MCP protocol

**Affected Stories**: P1 (MCP Integration)

---

## 7. Key Entities

```typescript
// Core configuration types
interface SiteConfig {
  id: string;              // Unique identifier (kebab-case)
  name: string;            // Human-readable name
  type: SiteType;          // marketing | docs | app | landing
  brand: BrandConfig;      // Design tokens
  stack: StackConfig;      // Technology choices
  pages: Record<string, PageConfig>;
  workflow?: string;       // Named workflow or inline steps
}

interface BrandConfig {
  palette: string;         // ocean | emerald | sunset | custom
  font: string;            // inter | geist | system
  logo?: string;           // Path to logo asset
}

interface PageConfig {
  layout: string;          // Compressed notation: hero+features+cta
  source?: string;         // Content source path
}

// Full schema in schema/sites-schema.yaml
```

---

## 8. Out of Scope (Explicit)

The following are explicitly NOT part of v1 (Phase 0):

| Feature | Reason | Phase |
|---------|--------|-------|
| Visual builder sync (Builder.io) | Complexity + unvalidated demand | Phase 2 |
| Multi-plane view system | Unvalidated need + 4 weeks work | Phase 2 |
| Advanced workflows (fork/wait-all) | Nice-to-have, not essential | Phase 1 |
| AI content generation | Separate concern | Phase 3 |
| Deployment integration | Users can deploy manually | Phase 1 |
| GUI dashboard | CLI-first principle | Phase 3+ |
| Real-time collaboration | Enterprise feature | Future |
| Per-site git repos (polyrepo) | Monorepo first | Future |

---

## 9. Open Questions

| ID | Question | Status | Blocking? |
|----|----------|--------|-----------|
| Q1 | Monorepo (Turborepo) vs. polyrepo output? | **DECIDED: Monorepo** | No |
| Q2 | Default framework (Next.js App Router)? | NEEDS INPUT | Yes |
| Q3 | Default UI kit (shadcn/ui)? | NEEDS INPUT | Yes |
| Q4 | Should sites share node_modules? | NEEDS RESEARCH | No |
| Q5 | How to handle template version updates? | DEFER to Phase 1 | No |

---

## 10. Approval Checklist

### Specification Phase Gate

- [ ] All P1 stories have acceptance tests
- [ ] No `[NEEDS CLARIFICATION]` markers remain
- [ ] Assumptions documented with validation methods
- [ ] Three-plane analysis complete
- [ ] Out-of-scope explicitly listed
- [ ] Open questions resolved or marked DEFER

### Approvers

| Role | Status | Date |
|------|--------|------|
| User (Product Owner) | PENDING | |
| Technical Lead | PENDING | |

---

*Proceed to plan.md only after specification approval.*
