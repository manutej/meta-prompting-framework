# Research: Schema Validation Libraries

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-002

---

## Executive Summary

**Recommendation**: Zod v4 for MPG configuration validation

Zod provides TypeScript-first validation with zero dependencies, excellent error messages, and 7x performance improvement in v4. Single source of truth for types and runtime validation.

---

## Comparison Matrix

| Library | Bundle Size | Performance | TS Integration | Best For |
|---------|-------------|-------------|----------------|----------|
| **Zod v4** | 5.36kB | Fast (7x v3) | Excellent | MPG ✓ |
| **Valibot** | 5.3kB | 2x Zod | Excellent | Bundle-conscious |
| **AJV** | Optimizable | Fastest | Good | JSON Schema |
| **TypeBox** | Small | Very Fast | Excellent | OpenAPI |
| **Arktype** | Optimized | 20x Zod | Perfect | Performance-critical |
| **Yup** | 12.2kB | Slow | Good | React forms |
| **io-ts** | Large | Moderate | Excellent | FP teams |

---

## Performance Benchmarks

| Library | Relative Speed | Notes |
|---------|---------------|-------|
| Arktype | 20x faster than Zod | Newest, less ecosystem |
| AJV | 11x faster than TypeBox | JSON Schema standard |
| Valibot | 2x faster than Zod | Lightweight alternative |
| Zod v4 | 7x faster than Zod v3 | Major improvement |
| Yup | 2000x slower than Arktype | Legacy, declining |

---

## Detailed Analysis

### Zod v4 (Recommended)

**Pros:**
- Zero dependencies
- TypeScript types inferred from schema (`z.infer<typeof schema>`)
- 7x performance improvement in v4 (May 2025)
- Excellent error messages with `zod-validation-error`
- Works with tRPC, React Hook Form, Next.js
- Standard Schema initiative member

**Cons:**
- Larger than Valibot (5.36kB vs <600B tree-shaken)
- Not as fast as AJV/Arktype for pure validation

**YAML Config Example:**
```typescript
const siteSchema = z.object({
  id: z.string().regex(/^[a-z][a-z0-9-]*$/),
  name: z.string(),
  type: z.enum(['marketing', 'docs', 'app', 'landing']),
  brand: z.object({
    palette: z.enum(['ocean', 'emerald', 'sunset']),
    font: z.string().default('inter')
  })
});

type SiteConfig = z.infer<typeof siteSchema>;
```

### Valibot (Alternative)

**Pros:**
- 90-95% smaller than Zod
- Tree-shakable to <600 bytes
- 2x faster runtime

**Cons:**
- Limited documentation
- Smaller community
- Functional API (different style)

**When to choose:** Bundle size is critical constraint

### AJV + TypeBox (Alternative)

**Pros:**
- Fastest validation
- JSON Schema standard
- Share schemas across languages

**Cons:**
- Two-step process
- Less TypeScript-native
- Requires compilation

**When to choose:** Multi-language microservices, OpenAPI generation

---

## Standard Schema Initiative (2025)

Unified interface between Zod, Valibot, and ArkType:
- Created by library authors
- ~60 lines of TypeScript types
- Adopted by tRPC, TanStack, Hono
- Enables tool integration without library-specific adapters

---

## Recommendation for MPG

### Choice: Zod v4

**Rationale:**
1. Single source of truth (schema = types)
2. Zero dependencies (safe for any deployment)
3. Excellent YAML + env var validation support
4. Standard in Next.js/tRPC ecosystem
5. Best developer experience for config validation

### Implementation Pattern:
```typescript
// Define once, validates both type and runtime
const configSchema = z.object({
  version: z.enum(['v1', 'v2']),
  sites: z.array(siteSchema),
  defaults: siteSchema.partial().optional()
});

// Load and validate
const config = configSchema.parse(await loadYamlConfig());
```

---

## Decision

**CONFIRMED: Zod v4**

Rationale:
1. Best TypeScript integration
2. Zero dependencies
3. 7x performance improvement in v4
4. Ecosystem standard (tRPC, Next.js)
5. Excellent error messages

---

## Sources

- [Zod v4 Release - InfoQ](https://www.infoq.com/news/2025/08/zod-v4-available/)
- [Zod vs Yup vs TypeBox - DEV Community](https://dev.to/dataformathub/zod-vs-yup-vs-typebox)
- [Standard Schema Official](https://standardschema.dev/)
- [Valibot - Lightweight Zod Alternative](https://valibot.dev/)
