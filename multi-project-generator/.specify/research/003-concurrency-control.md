# Research: Concurrency Control Libraries

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-003

---

## Executive Summary

**Recommendation**: p-queue for MPG parallel site generation

p-queue provides priority support, pause/resume, and concurrency control without Redis dependency. Perfect for 10-50 concurrent site generations.

---

## Comparison Matrix

| Library | Persistence | Dependencies | API | Best For |
|---------|-------------|--------------|-----|----------|
| **p-queue** | None | None (ESM) | Easy | MPG ✓ |
| **p-limit** | None | None | Easy | Simple throttling |
| **BullMQ** | Redis | IORedis | Medium | Production queues |
| **Bee-Queue** | Redis | Redis | Easy | High throughput |
| **Bottleneck** | Optional Redis | Optional | Easy | Rate limiting |
| **async.queue** | None | async lib | Medium | Legacy callback |

---

## Feature Comparison

| Feature | p-limit | p-queue | BullMQ | Bee-Queue |
|---------|---------|---------|--------|-----------|
| Priority queue | No | Yes | Yes | No |
| Pause/resume | No | Yes | Yes | Yes |
| Concurrency control | Yes | Yes | Yes | Yes |
| Rate limiting | No | Yes | Yes | No |
| Persistence | No | No | Yes | Yes |
| Retry built-in | No | No | Yes | Basic |
| Progress tracking | No | Events | Excellent | Basic |
| npm downloads | 171M/week | 8M/week | 2.3M/week | 38K/week |

---

## Detailed Analysis

### p-queue (Recommended)

**Pros:**
- Priority queue support (generate critical sites first)
- Pause/resume for graceful handling
- Concurrency AND rate limiting (intervalCap)
- onIdle() for completion detection
- No Redis required
- 8M+ weekly downloads

**Cons:**
- No persistence (lost on crash)
- ESM only
- No built-in retries (use with p-retry)

**Example:**
```typescript
import PQueue from 'p-queue';

const queue = new PQueue({ concurrency: 10 });

// Add sites with priority
sites.forEach(site => {
  queue.add(
    () => generateSite(site),
    { priority: site.priority }
  );
});

await queue.onIdle();
```

### p-limit (Simpler Alternative)

**Pros:**
- Minimal API (one function)
- 171M+ weekly downloads
- Lightweight

**Cons:**
- No priority support
- No pause/resume
- Basic concurrency only

**When to choose:** Very simple throttling needs

### BullMQ (Enterprise Alternative)

**Pros:**
- Full persistence (survives crashes)
- Distributed across servers
- Excellent retry with exponential backoff
- Progress tracking via job.progress()
- Dead Letter Queue support

**Cons:**
- Requires Redis
- More complex setup
- Overkill for MVP

**When to choose:** 100+ sites, distributed execution, crash recovery critical

---

## Error Handling Patterns

### With p-queue + p-retry:
```typescript
import PQueue from 'p-queue';
import pRetry from 'p-retry';

const queue = new PQueue({ concurrency: 10 });

const generateWithRetry = (site) => pRetry(
  () => generateSite(site),
  {
    retries: 3,
    onFailedAttempt: error => {
      console.log(`Attempt ${error.attemptNumber} failed for ${site.id}`);
    }
  }
);

sites.forEach(site => queue.add(() => generateWithRetry(site)));
```

---

## Performance Characteristics

### In-Memory (p-limit/p-queue)
- Handles 1000+ tasks easily for short-lived work
- Minimal overhead
- GC pressure increases over time
- Not for persistent workloads

### Redis-Backed (BullMQ/Bee-Queue)
- Scales to 100k+ jobs
- Network latency (50-100ms per operation)
- Survives crashes
- Distributed execution possible

---

## Recommendation for MPG

### Phase 0 (MVP): p-queue

**Rationale:**
1. Priority support for critical sites
2. Pause/resume for graceful handling
3. No Redis dependency (simpler deployment)
4. Sufficient for 10-50 concurrent sites
5. Can add p-retry for error recovery

### Phase 2+ (Scale): Consider BullMQ

**When:**
- Distributed generation across servers needed
- Crash recovery is critical
- 100+ concurrent sites

---

## Decision

**UPDATED: p-queue (from p-limit)**

Rationale:
1. Priority queue enables critical-first generation
2. Pause/resume for graceful shutdown
3. Rate limiting (intervalCap) prevents API overload
4. Same zero-dependency model as p-limit
5. onIdle() simplifies completion handling

---

## Sources

- [p-queue GitHub](https://github.com/sindresorhus/p-queue)
- [BullMQ Documentation](https://docs.bullmq.io/)
- [Bee-Queue Redis-based Queue](https://redis.io/blog/bee-queue-redis-based-distributed-queue/)
- [Modern JavaScript Concurrency 2025](https://dev.to/gkoos/modern-javascript-concurrency-2025-edition-h84)
