# Architectural Decision Records (ADRs)

**Document**: Records all significant architectural decisions for MPG
**Format**: [ADR Template by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)

---

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| ADR-001 | Monorepo Output Structure | ACCEPTED | 2026-01-04 |
| ADR-002 | Zod for Schema Validation | ACCEPTED | 2026-01-04 |
| ADR-003 | p-limit for Concurrency Control | ACCEPTED | 2026-01-04 |
| ADR-004 | Commander.js for CLI Framework | ACCEPTED | 2026-01-04 |
| ADR-005 | Template as File Copy + Substitution | ACCEPTED | 2026-01-04 |
| ADR-006 | MCP SDK for Agent Integration | ACCEPTED | 2026-01-04 |
| ADR-007 | Compact Workflow DSL with + Notation | ACCEPTED | 2026-01-04 |
| ADR-008 | Three-Plane Analysis for Spec Validation | ACCEPTED | 2026-01-04 |

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
Use **Turborepo monorepo** for all generated sites.

### Rationale
- **Shared dependencies**: Reduces disk usage and install time
- **Atomic updates**: Change shared UI, all sites update
- **Unified CI/CD**: One pipeline for all sites
- **Proven at scale**: Turborepo handles 100+ packages

### Consequences
**Positive**:
- Faster builds via Turborepo caching
- Easier dependency management
- Single source of truth

**Negative**:
- Large repos can be slow to clone
- Permission management more complex
- Single point of failure for CI

### Alternatives Considered
- **Polyrepo**: More flexible but coordination overhead for 10+ sites
- **No orchestration**: Too manual, doesn't scale

---

## ADR-002: Zod for Schema Validation

### Status
ACCEPTED

### Context
YAML configuration must be validated at runtime. Need both:
- Runtime validation (catch errors before execution)
- TypeScript types (compile-time safety)

### Decision
Use **Zod** for schema definition and validation.

### Rationale
- **Single source of truth**: Schema generates TypeScript types
- **Better errors**: Human-readable validation messages
- **Ecosystem**: Works with tRPC, React Hook Form, etc.
- **Runtime + compile-time**: Validates at both levels

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
- **JSON Schema only**: No TypeScript inference
- **io-ts**: More complex API, steeper learning curve
- **Yup**: Less TypeScript-native, weaker inference
- **TypeBox**: Good but less ecosystem support

---

## ADR-003: p-limit for Concurrency Control

### Status
ACCEPTED

### Context
Generating 10-20 sites in parallel requires concurrency control to:
- Prevent resource exhaustion
- Enable configurable parallelism
- Handle failures gracefully

### Decision
Use **p-limit** library for concurrency control.

### Rationale
- **Minimal API**: One function, easy to understand
- **Proven**: 50M+ weekly downloads
- **No dependencies**: Small footprint
- **Sufficient for MVP**: No need for persistent queues

### Consequences
**Positive**:
- Simple to implement and test
- Low overhead
- Easy to reason about

**Negative**:
- No persistence (jobs lost on crash)
- No distributed execution
- No built-in retry (must implement)

### Alternatives Considered
- **Bull/BullMQ**: Overkill for MVP, requires Redis
- **p-queue**: More features but more complexity
- **Custom implementation**: NIH risk, bugs likely

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

### Rationale
- **Most popular**: Battle-tested, well-documented
- **Simple API**: Declarative command definition
- **Built-in help**: Auto-generates usage text
- **Flexible**: Supports our verb+target+modifiers pattern

### Consequences
**Positive**:
- Quick to implement
- Good documentation
- Large community

**Negative**:
- Less opinionated (more decisions needed)
- No built-in config file support
- Manual output formatting

### Alternatives Considered
- **oclif**: Too heavy for MVP, enterprise-focused
- **yargs**: More complex API
- **citty**: Newer, less proven, but promising
- **CAC**: Lightweight but less features

---

## ADR-005: Template as File Copy + Substitution

### Status
ACCEPTED

### Context
Need to scaffold sites from templates with:
- Variable substitution (site name, colors, etc.)
- File renaming (patterns like `__name__.tsx`)
- Any file type support

### Decision
Use **file copy + string substitution** pattern.

### Implementation
```
1. Copy template directory to output
2. Walk all files recursively
3. Replace {{variable}} patterns in file contents
4. Rename files with __pattern__ in names
```

### Rationale
- **Simple**: Easy to understand and debug
- **Universal**: Works with any file type
- **Transparent**: Templates look like real projects
- **No DSL**: Standard file structure

### Consequences
**Positive**:
- Easy to create new templates
- Templates are valid projects themselves
- No special syntax to learn

**Negative**:
- Less powerful than full template engines
- Variable syntax visible in templates
- No conditionals or loops

### Alternatives Considered
- **EJS/Handlebars**: More complex, overkill for MVP
- **Plop**: Good but separate tool
- **Hygen**: Another dependency, learning curve

---

## ADR-006: MCP SDK for Agent Integration

### Status
ACCEPTED

### Context
Claude Code and other AI agents need programmatic access to MPG via Model Context Protocol (MCP).

### Decision
Use **@modelcontextprotocol/sdk** to expose CLI commands as MCP tools.

### Rationale
- **Official SDK**: Maintained by Anthropic
- **Standard protocol**: Works with any MCP client
- **1:1 mapping**: Each CLI command becomes an MCP tool
- **Constitution Principle**: "MCP-Native Features"

### Consequences
**Positive**:
- AI agents can orchestrate site generation
- Structured input/output via JSON Schema
- Progress streaming support

**Negative**:
- MCP protocol still evolving
- Additional server process to run
- Testing requires MCP client

### Alternatives Considered
- **REST API**: More familiar but not AI-native
- **gRPC**: Overkill, not needed for this use case
- **Custom protocol**: NIH, fragmentation

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
```

---

*ADRs are immutable once accepted. To change a decision, create a new ADR that supersedes the old one.*
