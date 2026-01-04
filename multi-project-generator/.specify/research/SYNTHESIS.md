# Research Synthesis: MPG Technology Stack

**Date**: 2026-01-04
**Status**: Complete
**Purpose**: Consolidate all ADR research into actionable recommendations

> **HIGH PRIORITY**: See [000-core-technologies.md](./000-core-technologies.md) for up-to-date documentation on all core libraries (versions, APIs, breaking changes).

---

## Executive Summary

After comprehensive research across 6 architectural decisions, the following technology stack is recommended for MPG:

| Layer | Technology | Confidence | Source |
|-------|------------|------------|--------|
| **Monorepo** | Turborepo + pnpm | HIGH | ADR-001 |
| **Validation** | Zod v4 | HIGH | ADR-002 |
| **Concurrency** | p-queue (UPDATED) | HIGH | ADR-003 |
| **CLI** | Commander.js | HIGH | ADR-004 |
| **Templating** | giget + Handlebars | HIGH | ADR-005 |
| **AI Integration** | MCP Protocol | HIGH | ADR-006 |

---

## Key Research Findings

### 1. Monorepo Tools (ADR-001)

**Winner: Turborepo + pnpm workspaces**

| Metric | Turborepo | Nx | Lerna |
|--------|-----------|-----|-------|
| Build Speed | 2.8s | 8.3s | 44.8s |
| Setup Time | 15 min | 2.5+ hrs | 30 min |
| Config Lines | ~20 | 200+ | 50+ |

**Key Insight**: Turborepo is 3x faster than Nx for our scale (10-50 sites). Nx only becomes necessary at 100+ sites with distributed CI/CD requirements.

---

### 2. Schema Validation (ADR-002)

**Winner: Zod v4**

| Library | Bundle | Performance | TS Integration |
|---------|--------|-------------|----------------|
| Zod v4 | 5.36kB | 7x faster than v3 | Excellent |
| Valibot | <600B | 2x faster | Excellent |
| AJV | Varies | Fastest | Good |

**Key Insight**: Zod v4 provides single source of truth for types AND runtime validation. Standard Schema initiative (2025) ensures interoperability.

---

### 3. Concurrency Control (ADR-003)

**Winner: p-queue (UPDATED from p-limit)**

| Feature | p-limit | p-queue | BullMQ |
|---------|---------|---------|--------|
| Priority Queue | ❌ | ✅ | ✅ |
| Pause/Resume | ❌ | ✅ | ✅ |
| Rate Limiting | ❌ | ✅ | ✅ |
| Redis Required | ❌ | ❌ | ✅ |

**Key Insight**: p-queue adds priority support (generate critical sites first) and pause/resume capabilities without adding Redis dependency. Same author (Sindre Sorhus), similar API.

**UPDATE REQUIRED**: ADR-003 should change from p-limit to p-queue.

---

### 4. CLI Framework (ADR-004)

**Winner: Commander.js**

| Framework | Stars | Learning | Best For |
|-----------|-------|----------|----------|
| Commander | 26.2k | Easy | 5-20 commands ✓ |
| oclif | 8.9k | Medium | Enterprise |
| yargs | 11k | Medium | Complex args |

**Key Insight**: Commander.js is the perfect fit for MPG's 10 commands. oclif becomes valuable at 20+ commands or when plugins are needed.

---

### 5. Template Systems (ADR-005)

**Winner: Hybrid (giget + Handlebars + custom orchestrator)**

| Tool | Type | Batch-Ready | Best For |
|------|------|-------------|----------|
| giget | Download | Yes | Template fetch |
| Handlebars | Engine | Component | Variable substitution |
| Plop | Generator | No | Interactive (not MPG) |

**Key Insight**: No single tool handles batch generation. The hybrid approach combines:
1. **giget** for parallel template downloads
2. **Handlebars** for safe `{{variable}}` substitution
3. **Custom orchestrator** for 10-100 site parallelism

**UPDATE REQUIRED**: ADR-005 should be refined to explicitly specify the hybrid approach.

---

### 6. AI Integration (ADR-006)

**Winner: MCP (Model Context Protocol)**

| Protocol | AI-Native | Ecosystem | Cross-Vendor |
|----------|-----------|-----------|--------------|
| MCP | ⭐⭐⭐⭐⭐ | 8M+ downloads | ✅ OpenAI, Google, MS |
| REST | ⭐⭐ | Universal | N/A |
| gRPC | ⭐⭐⭐ | Enterprise | No |

**Key Insight**: MCP won the "M×N problem" - universal protocol eliminates integration explosion. March 2025 OpenAI adoption and December 2025 Linux Foundation governance solidify it as the standard.

---

## Technology Stack Summary

### Phase 0 (MVP) - All Confirmed

```
┌─────────────────────────────────────────────────────┐
│                   MPG CLI (Commander.js)            │
├─────────────────────────────────────────────────────┤
│  Config Validation (Zod v4)                         │
├─────────────────────────────────────────────────────┤
│  Template Download (giget) → Process (Handlebars)   │
├─────────────────────────────────────────────────────┤
│  Parallel Execution (p-queue)                       │
├─────────────────────────────────────────────────────┤
│  Monorepo Output (Turborepo + pnpm)                 │
├─────────────────────────────────────────────────────┤
│  AI Integration (MCP Server)                        │
└─────────────────────────────────────────────────────┘
```

### Phase 2+ (Scale) - Migration Paths

| Current | Scale Trigger | Migration |
|---------|---------------|-----------|
| Turborepo | 100+ sites, >5min CI | → Nx |
| p-queue | Distributed, crash recovery | → BullMQ |
| Commander | 20+ commands, plugins | → oclif |

---

## ADR Updates Required

Based on research, the following ADRs need updates:

### ADR-003: p-limit → p-queue
- **Reason**: Priority queue and pause/resume are valuable for site generation
- **Impact**: Minimal - similar API, same author

### ADR-005: Clarify hybrid approach
- **Reason**: Original ADR mentions "file copy + substitution" but research shows giget + Handlebars is optimal
- **Impact**: None - clarification only

---

## Risk Assessment

| Risk | Mitigation | Confidence |
|------|------------|------------|
| Turborepo limitations at scale | Documented migration path to Nx | HIGH |
| MCP protocol changes | Using official SDK, Linux Foundation governance | HIGH |
| p-queue crashes losing jobs | Add p-retry, consider BullMQ for Phase 2 | MEDIUM |
| Template complexity | Handlebars is logic-less, safe by design | HIGH |

---

## Recommendation

**PROCEED WITH IMPLEMENTATION**

All 6 ADRs have been validated through research. The recommended stack is:
- Well-documented
- Widely adopted
- Appropriate for MPG scale
- Has clear migration paths for growth

Update ADR-003 and ADR-005 to reflect research findings, then proceed to Phase 0 implementation.

---

## Sources

All research files are available in `.specify/research/`:
- [001-monorepo-tools.md](./001-monorepo-tools.md)
- [002-schema-validation.md](./002-schema-validation.md)
- [003-concurrency-control.md](./003-concurrency-control.md)
- [004-cli-frameworks.md](./004-cli-frameworks.md)
- [005-template-systems.md](./005-template-systems.md)
- [006-agent-integration.md](./006-agent-integration.md)
