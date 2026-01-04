# Architectural Decision Records (ADRs)

**Document**: Records all significant architectural decisions for MPG
**Format**: [ADR Template by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
**Research**: All decisions validated via `.specify/research/` analysis

---

## ADR Index

| ID | Title | Status | Date | Research |
|----|-------|--------|------|----------|
| ADR-001 | Monorepo Output Structure | ACCEPTED | 2026-01-04 | [001-monorepo-tools](../../research/001-monorepo-tools.md) |
| ADR-002 | Zod for Schema Validation | ACCEPTED | 2026-01-04 | [002-schema-validation](../../research/002-schema-validation.md) |
| ADR-003 | p-queue for Concurrency Control | ACCEPTED | 2026-01-04 | [003-concurrency-control](../../research/003-concurrency-control.md) |
| ADR-004 | Commander.js for CLI Framework | ACCEPTED | 2026-01-04 | [004-cli-frameworks](../../research/004-cli-frameworks.md) |
| ADR-005 | Hybrid Template System | ACCEPTED | 2026-01-04 | [005-template-systems](../../research/005-template-systems.md) |
| ADR-006 | MCP SDK for Agent Integration | ACCEPTED | 2026-01-04 | [006-agent-integration](../../research/006-agent-integration.md) |
| ADR-007 | Compact Workflow DSL with + Notation | ACCEPTED | 2026-01-04 | N/A |
| ADR-008 | Three-Plane Analysis for Spec Validation | ACCEPTED | 2026-01-04 | N/A |

---

## ADR-001: Monorepo Output Structure

### Status
ACCEPTED

### Context
Generated sites need a home. Options:
1. **Monorepo**: All sites in one repository using Turborepo
2. **Polyrepo**: Each site gets its own repository
3. **Hybrid**: Monorepo for code, separate repos for content

### Decision
Use **Turborepo + pnpm workspaces** for all generated sites.

### Research Justification

**Benchmarks (MacBook Pro M2, 16GB RAM, 10 packages):**

| Tool | Build Time | vs Baseline |
|------|-----------|-------------|
| Turborepo | 2.8s | 3x faster |
| Nx | 8.3s | 1x baseline |
| Lerna (alone) | 44.8s | 16x slower |

**Setup Comparison:**
- Turborepo: 15 minutes, ~20 lines config
- Nx: 2.5+ hours, 200+ lines config

### Rationale
1. **3x faster builds** than Nx for our scale (10-50 sites)
2. **15-minute setup** vs 2.5+ hours for Nx
3. **Perfect Vercel integration** for deployment
4. **Shared dependencies**: Reduces disk usage and install time
5. **Atomic updates**: Change shared UI, all sites update
6. **Migration path**: Can move to Nx at 100+ sites if needed

### Consequences
**Positive**:
- Faster builds via Turborepo caching
- Easier dependency management
- Single source of truth
- 40-60% disk savings with pnpm

**Negative**:
- Large repos can be slow to clone
- Permission management more complex
- Single point of failure for CI

### Alternatives Considered
- **Nx**: Overkill for MVP, 2.5+ hour setup, needed for 100+ sites
- **Lerna**: 16x slower builds, versioning-focused
- **Rush**: Microsoft-scale (1000+ packages), complex
- **Moon**: Emerging, less ecosystem support
- **Polyrepo**: Coordination overhead for 10+ sites

---

## ADR-002: Zod for Schema Validation

### Status
ACCEPTED

### Context
YAML configuration must be validated at runtime. Need both:
- Runtime validation (catch errors before execution)
- TypeScript types (compile-time safety)

### Decision
Use **Zod v4** for schema definition and validation.

### Research Justification

**Performance Comparison:**

| Library | Bundle Size | Performance | TS Integration |
|---------|-------------|-------------|----------------|
| Zod v4 | 5.36kB | 7x faster than v3 | Excellent |
| Valibot | <600B (tree-shaken) | 2x faster than Zod | Excellent |
| AJV | Varies | Fastest | Good |
| Arktype | Optimized | 20x faster than Zod | Perfect |

**Why Zod over faster alternatives:**
- Standard Schema initiative member (interoperable)
- Ecosystem standard (tRPC, Next.js, React Hook Form)
- Zero dependencies
- Zod v4 (May 2025) closed the performance gap

### Rationale
1. **Single source of truth**: `z.infer<typeof schema>` generates types
2. **7x performance improvement** in v4 vs v3
3. **Zero dependencies**: Safe for any deployment
4. **Ecosystem standard**: Works with tRPC, React Hook Form, etc.
5. **Best developer experience** for config validation

### Consequences
**Positive**:
- Types always match validation
- Excellent error messages with paths
- Composable schemas

**Negative**:
- Additional dependency
- Schema must be maintained alongside types
- Learning curve for team

### Alternatives Considered
- **Valibot**: 90-95% smaller, but smaller ecosystem
- **AJV + TypeBox**: Faster, but two-step process, less TS-native
- **Arktype**: 20x faster, but newest, less proven
- **Yup**: 2000x slower than Arktype, declining

---

## ADR-003: p-queue for Concurrency Control

### Status
ACCEPTED (Updated from p-limit)

### Context
Generating 10-20 sites in parallel requires concurrency control to:
- Prevent resource exhaustion
- Enable configurable parallelism
- Handle failures gracefully
- **NEW**: Prioritize critical sites

### Decision
Use **p-queue** library for concurrency control.

### Research Justification

**Feature Comparison:**

| Feature | p-limit | p-queue | BullMQ |
|---------|---------|---------|--------|
| Priority Queue | ❌ | ✅ | ✅ |
| Pause/Resume | ❌ | ✅ | ✅ |
| Rate Limiting | ❌ | ✅ | ✅ |
| onIdle() | ❌ | ✅ | ✅ |
| Redis Required | ❌ | ❌ | ✅ |
| Weekly Downloads | 171M | 8M | 2.3M |

**Why upgraded from p-limit:**
- Same author (Sindre Sorhus), similar API
- Priority queue enables "generate critical sites first"
- Pause/resume for graceful shutdown
- intervalCap for rate limiting (prevent API overload)
- onIdle() simplifies completion detection

### Rationale
1. **Priority support**: Generate critical sites first
2. **Pause/resume**: Graceful handling during shutdown
3. **Rate limiting**: Prevent API overload with intervalCap
4. **No Redis**: Simpler deployment than BullMQ
5. **8M+ weekly downloads**: Well-maintained

### Consequences
**Positive**:
- Simple to implement and test
- Priority-based execution
- Graceful pause/resume
- No infrastructure dependencies

**Negative**:
- No persistence (jobs lost on crash)
- No distributed execution
- No built-in retry (use with p-retry)

### Alternatives Considered
- **p-limit**: Simpler but no priority, no pause/resume
- **BullMQ**: Full-featured but requires Redis, overkill for MVP
- **Bee-Queue**: Redis-based, no priority queue
- **Bottleneck**: Rate limiting focused, less queue management

### Migration Path
At 100+ sites or when crash recovery is critical, consider BullMQ.

---

## ADR-004: Commander.js for CLI Framework

### Status
ACCEPTED

### Context
Need CLI framework for `/mpg` commands that supports:
- Subcommands (list, plan, apply, etc.)
- Options and flags
- Help generation
- Extensibility

### Decision
Use **Commander.js** for CLI implementation.

### Research Justification

**Framework Comparison:**

| Framework | GitHub Stars | TypeScript | Learning Curve | Best For |
|-----------|-------------|------------|----------------|----------|
| Commander | 26.2k | Good | Easy (hours) | 5-20 commands ✓ |
| oclif | 8.9k | Native | Medium (days) | Enterprise |
| yargs | 11k | Good | Medium | Complex args |
| citty | 878 | Good | Easy | UnJS ecosystem |
| CAC | Varies | Native | Easy | Minimal |

**Why Commander over oclif:**
- MPG has ~10 commands (Commander sweet spot)
- oclif requires 2-3 day setup vs hours
- oclif plugins not needed for MVP

### Rationale
1. **Most popular**: 26.2k stars, battle-tested
2. **Perfect fit**: 10 commands is Commander sweet spot
3. **Easy to learn**: Hours, not days
4. **Excellent help generation**: Auto-generates usage
5. **Flexible**: Supports verb+target+modifiers pattern

### Consequences
**Positive**:
- Quick to implement
- Good documentation
- Large community

**Negative**:
- Less opinionated (more decisions needed)
- No built-in config file support
- Manual output formatting (use Chalk)

### Alternatives Considered
- **oclif**: Too heavy for MVP, 2-3 day setup, better at 20+ commands
- **yargs**: More complex API, dot-notation overkill
- **citty**: Newer, less proven, but promising for UnJS
- **CAC**: Zero deps but less features

### Migration Path
If MPG grows to 20+ commands or needs plugins, consider oclif.

---

## ADR-005: Hybrid Template System

### Status
ACCEPTED (Clarified from "file copy + substitution")

### Context
Need to scaffold sites from templates with:
- Variable substitution (site name, colors, etc.)
- File renaming (patterns like `__name__.tsx`)
- Any file type support
- **Batch generation** of 10-100 sites in parallel

### Decision
Use **hybrid approach**: giget + Handlebars + custom orchestrator.

### Research Justification

**Tool Analysis:**

| Tool | Type | Batch-Ready | Best For |
|------|------|-------------|----------|
| giget | Download | Yes (parallel) | Template fetch |
| Handlebars | Engine | Component | Variable substitution |
| Plop | Generator | No (interactive) | Single component |
| Hygen | Generator | Partial | Code injection |
| Yeoman | Framework | No (interactive) | Enterprise legacy |

**Key insight**: No single tool handles batch generation. All modern framework CLIs (Next.js, Remix, Astro) use interactive single-project CLIs.

**Hybrid Pipeline:**
```
1. Download (giget) → Fast parallel tarball downloads
2. Substitute (Handlebars) → Safe {{variable}} syntax
3. Rename (custom) → __pattern__ in filenames
4. Orchestrate (p-queue) → 10-50 concurrent sites
```

### Implementation
```typescript
import { downloadTemplate } from 'giget';
import Handlebars from 'handlebars';
import PQueue from 'p-queue';

async function generateSites(sites: SiteConfig[], concurrency = 10) {
  const queue = new PQueue({ concurrency });

  for (const site of sites) {
    queue.add(async () => {
      // 1. Download template (giget)
      await downloadTemplate(`github:org/template`, { dir: site.outDir });

      // 2. Process text files (Handlebars)
      for (const file of await glob(`${site.outDir}/**/*.{ts,tsx,json}`)) {
        const template = Handlebars.compile(await readFile(file, 'utf-8'));
        await writeFile(file, template(site));
      }

      // 3. Rename __pattern__ files
      // ...
    });
  }

  await queue.onIdle();
}
```

### Rationale
1. **giget**: Fast parallel downloads (4.2M weekly downloads)
2. **Handlebars**: Logic-less = safe (no XSS), familiar syntax
3. **Custom orchestrator**: Enables 10-100 site parallelism
4. **Minimal dependencies**: No heavy frameworks
5. **Binary-safe**: giget downloads, Handlebars only touches text

### Consequences
**Positive**:
- Easy to create new templates
- Templates are valid projects themselves
- Full control over generation
- Binary files handled correctly

**Negative**:
- More code to maintain than single-tool solution
- No built-in conditionals (use separate templates)
- Custom parser for __pattern__ filenames

### Alternatives Considered
- **Plop only**: Interactive, not batch-ready
- **Hygen only**: EJS security concerns, no batch orchestration
- **Yeoman**: Heavy, dated, interactive-only
- **degit**: Older, less features than giget

---

## ADR-006: MCP SDK for Agent Integration

### Status
ACCEPTED

### Context
Claude Code and other AI agents need programmatic access to MPG via Model Context Protocol (MCP).

### Decision
Use **@modelcontextprotocol/sdk** to expose CLI commands as MCP tools.

### Research Justification

**Protocol Comparison:**

| Protocol | AI-Native | Streaming | Ecosystem | Best For |
|----------|-----------|-----------|-----------|----------|
| MCP | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8M+ downloads | MPG ✓ |
| REST | ⭐⭐ | ⭐⭐ | Universal | Legacy |
| GraphQL | ⭐⭐⭐ | ⭐⭐ | Large | Data exploration |
| gRPC | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Enterprise | Internal services |

**MCP Adoption (2024-2025):**
- Nov 2024: Anthropic launches MCP
- Mar 2025: OpenAI announces support (Sam Altman tweet)
- Apr 2025: 8M+ downloads, 5,800+ servers
- Dec 2025: Linux Foundation governance (AAIF)

**The M×N Problem Solved:**
- Before MCP: 10 AI providers × 100 tools = 1,000 integrations
- After MCP: 1 universal protocol

### Rationale
1. **Industry standard**: Unprecedented cross-vendor adoption (2025)
2. **Provider-agnostic**: Works with Claude, GPT, Gemini
3. **Tool discovery built-in**: Agents find available tools
4. **Streaming support**: Progress updates during generation
5. **85% token reduction**: Tool Search feature in Claude Code

### MCP Tool Definitions for MPG
```typescript
const tools = [
  {
    name: "list_sites",
    description: "List configured sites",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["marketing", "docs", "app"] },
        format: { type: "string", enum: ["summary", "full"] }
      }
    }
  },
  {
    name: "plan_sites",
    description: "Preview generation (dry run)",
    inputSchema: { /* ... */ }
  },
  {
    name: "apply_sites",
    description: "Execute site generation",
    inputSchema: { /* ... */ }
  }
];
```

### Consequences
**Positive**:
- AI agents can orchestrate site generation
- Structured input/output via JSON Schema
- Progress streaming support
- Future-proof (Linux Foundation governance)

**Negative**:
- MCP protocol still evolving
- Additional server process to run
- Testing requires MCP client

### Alternatives Considered
- **REST API**: More familiar but not AI-native, verbose for tools
- **gRPC**: Doesn't work in browsers, not AI-designed
- **Function Calling (OpenAI)**: Vendor lock-in
- **LangChain**: Python-focused, different paradigm

---

## ADR-007: Compact Workflow DSL with + Notation

### Status
ACCEPTED

### Context
Users need to define workflows (sequences of steps). Options:
1. Verbose YAML arrays
2. Compact string notation
3. Custom DSL file format

### Decision
Use **compact string notation** with `+` separator.

### Example
```yaml
# Instead of:
workflow:
  steps:
    - scaffold
    - design
    - generate-copy
    - deploy

# Use:
workflow: scaffold+design+generate-copy+deploy
```

### Rationale
- **Low keystrokes**: Faster to write
- **Readable**: Clear sequence indication
- **Familiar**: Similar to shell pipelines
- **Composable**: Named workflows can reference others

### Consequences
**Positive**:
- 80% reduction in YAML verbosity
- Easier to read at a glance
- Simple to parse (split on +)

**Negative**:
- Limited to sequential steps (use pipeline syntax for parallel)
- Step names cannot contain +
- Custom parser needed

### Alternatives Considered
- **YAML arrays only**: Too verbose for simple cases
- **Shell-like pipes**: `scaffold | design` less clear
- **Separate workflow file**: Adds complexity

---

## ADR-008: Three-Plane Analysis for Spec Validation

### Status
ACCEPTED

### Context
Specifications often suffer from:
- Scope creep (building too much)
- Unrealistic timelines
- Unclear priorities
- Missing stakeholder consideration

### Decision
Use **MERCURIO Three-Plane Analysis** for all major decisions.

### The Three Planes
1. **Mental**: Is this TRUE? (Evidence-based, intellectually sound)
2. **Physical**: Can we DO this? (Feasible, realistic, resourced)
3. **Spiritual**: Is this RIGHT? (Serves users, values-aligned)

### Rationale
- **Surfaces conflicts early**: Before implementation starts
- **Prevents over-engineering**: Physical plane reality check
- **Stakeholder focus**: Spiritual plane ensures user value
- **Documented decisions**: Future maintainers understand why

### Consequences
**Positive**:
- Better scoped specifications
- Conflicts resolved before coding
- Clear prioritization

**Negative**:
- Adds process overhead
- Requires discipline to apply
- May slow initial spec writing

### Alternatives Considered
- **Traditional requirements only**: Misses feasibility/values
- **PRD format**: Less structured risk analysis
- **No formal process**: Ad-hoc, inconsistent

---

## Template for New ADRs

```markdown
## ADR-XXX: [Title]

### Status
PROPOSED | ACCEPTED | DEPRECATED | SUPERSEDED

### Context
[What is the issue that we're seeing that is motivating this decision?]

### Decision
[What is the change that we're proposing and/or doing?]

### Research Justification
[Link to research file if applicable]
[Key benchmarks or comparisons]

### Rationale
[Why is this the best choice among alternatives?]

### Consequences
**Positive**:
- [Benefit 1]
- [Benefit 2]

**Negative**:
- [Drawback 1]
- [Drawback 2]

### Alternatives Considered
- [Alternative 1]: [Why not chosen]
- [Alternative 2]: [Why not chosen]

### Migration Path
[When and how to migrate if this decision needs to change]
```

---

*ADRs are immutable once accepted. To change a decision, create a new ADR that supersedes the old one.*
