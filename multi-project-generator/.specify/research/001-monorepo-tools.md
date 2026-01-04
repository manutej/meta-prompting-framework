# Research: Monorepo Tools Comparison

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-001

---

## Executive Summary

**Recommendation**: Turborepo + pnpm workspaces for MPG (10-100 sites)

Turborepo provides 3x faster builds than Nx with minimal configuration (20 lines vs 200+). For 10-50 sites, this is optimal. Nx becomes necessary only at 100+ sites requiring distributed CI/CD.

---

## Comparison Matrix

| Tool | GitHub Stars | Scale | Learning Curve | Best For |
|------|-------------|-------|----------------|----------|
| **Turborepo** | 28,384 | 10-50 packages | Easy (15 min) | MPG v1 ✓ |
| **Nx** | ~20,000 | 50-500+ packages | Hard (2.5+ hr) | Enterprise |
| **Lerna** | 36,000 | Versioning focus | Easy | NPM publishing |
| **Rush** | 152 | 1000+ packages | Medium | Microsoft scale |
| **pnpm workspaces** | 31,534 | Dependency only | Very Easy | Foundation layer |
| **Moon** | ~2,000 | Emerging | Medium | Polyglot |

---

## Performance Benchmarks

**MacBook Pro M2, 16GB RAM, 10 packages:**

| Tool | Build Time | vs Baseline |
|------|-----------|-------------|
| Turborepo | 2.8s | 3x faster |
| Nx | 8.3s | 1x baseline |
| Lerna (alone) | 44.8s | 16x slower |

---

## Detailed Analysis

### Turborepo (Recommended)

**Pros:**
- 3x faster builds than Nx
- Minimal config (~20 lines)
- 15-minute setup
- Vercel integration (perfect for deployment)
- Being rewritten in Rust (2025)

**Cons:**
- Single-machine only (no distributed execution)
- Limited to 50-100 packages comfortably
- No built-in versioning/publishing

**When to choose:** 10-50 sites, small-medium teams, fast iteration

### Nx

**Pros:**
- Distributed execution across 50+ machines
- Advanced dependency graph
- 50+ plugins
- Enterprise-grade

**Cons:**
- 2.5+ hour setup
- 200+ lines config typical
- Steep learning curve

**When to choose:** 100+ sites, enterprise teams, polyglot codebases

### pnpm workspaces

**Pros:**
- Built into pnpm (no extra tool)
- 40-60% disk savings
- Fastest package manager
- 19.9% market share (2024)

**Cons:**
- Only handles dependencies, not task orchestration
- Needs Turborepo or Nx on top

**When to choose:** Always use as foundation with Turborepo

---

## Recommendation for MPG

### Phase 0 (MVP): Turborepo + pnpm
```
pnpm init workspace
+ Turborepo (add in <15 min)
→ Deploy on Vercel
```

### Phase 2+ (Scale): Consider Nx
```
If 100+ sites AND build time > 5 min in CI:
→ Migrate to Nx with distributed execution
```

---

## Decision

**CONFIRMED: Turborepo + pnpm workspaces**

Rationale:
1. 3x faster builds for our scale (10-50 sites)
2. 15-minute setup vs 2.5+ hours for Nx
3. Perfect Vercel integration for deployment
4. Can migrate to Nx later if needed

---

## Sources

- [Nx vs Turborepo Comparison - Wisp CMS](https://www.wisp.blog/blog/nx-vs-turborepo)
- [Turborepo Official Docs](https://turborepo.com/docs)
- [Top 5 Monorepo Tools 2025 - Aviator Blog](https://www.aviator.co/blog/monorepo-tools/)
- [GitHub Benchmarks - vsavkin/large-monorepo](https://github.com/vsavkin/large-monorepo)
