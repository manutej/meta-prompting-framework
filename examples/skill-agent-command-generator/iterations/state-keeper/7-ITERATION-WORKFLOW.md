# 7-Iteration Meta-Prompting: State Keeper Agent

> Construct → Deconstruct → Reconstruct through 7 iterations

---

## Iteration 1: Foundation - Why Centralize State?

### Question
Why do we need a dedicated agent for state management?

### Analysis

**State Keeper** is a specialized agent that centralizes the 4-layer state management (context/local/artifacts/coordination) defined in the state-management skill.

**Problem with Distributed State:**
- Each agent managing its own coordination state = complexity
- Inconsistent state access patterns
- Hard to audit who accessed what
- Difficult to enforce policies (quotas, permissions)

**Benefits of Centralization:**
- **Single source of truth** - One agent owns coordination state
- **Consistent policies** - Enforce access control, rate limits
- **Observability** - Track all state operations
- **Transactions** - ACID guarantees across operations

**Analogy:**
```
Distributed State = Everyone manages own bank account
Centralized State = Bank manages all accounts (State Keeper)
```

### Key Insight
**Centralization trades distributed complexity for coordination cost.** Worth it when consistency and observability matter more than performance.

---

## Iteration 2: Pattern Extraction - Agent Responsibilities

### Question
What should a State Keeper agent do vs not do?

### Responsibility Analysis

**DOES:**
1. **Manage Coordination State** (Layer 4)
   - Locks, semaphores, transactions
   - Shared counters, flags
   - Workflow status

2. **Enforce Policies**
   - Access control (who can read/write what)
   - Rate limiting (max ops per agent)
   - Quota enforcement (state size limits)

3. **Provide Transactions**
   - Multi-operation atomicity
   - Isolation between concurrent transactions
   - Rollback on failure

4. **Audit Trail**
   - Log all state operations
   - Track who/when/what changed
   - Support debugging and compliance

**DOES NOT:**
1. **Manage Local State** - That's private to each agent
2. **Store Artifacts** - Append-only log is separate
3. **Manage Context** - Immutable config is read-only
4. **Execute Business Logic** - Just provides state storage

**Layer Responsibilities:**

| Layer | Owner | State Keeper Role |
|-------|-------|-------------------|
| Context (L1) | System | Read-only access |
| Local (L2) | Each agent | None (private) |
| Artifacts (L3) | Artifact store | None (append-only) |
| Coordination (L4) | **State Keeper** | **Manages** |

### Key Insight
**State Keeper owns coordination layer only.** Other layers handled by their respective owners.

---

## Iteration 3: Cognitive Load - Delegation Benefits

### Question
How does delegating to State Keeper reduce cognitive load?

### Load Analysis

**Before State Keeper (Each Agent Manages Own State):**
```
Agent Working Memory:
├─ Task logic: 3-4 slots
├─ State coordination: 3-4 slots (locks, transactions)
├─ Budget tracking: 1-2 slots
└─ Total: 7-10 slots ❌ OVERLOAD
```

**After State Keeper (Delegation):**
```
Agent Working Memory:
├─ Task logic: 3-4 slots
├─ State operations: 1 slot (just send request to State Keeper)
├─ Budget tracking: 1-2 slots
└─ Total: 5-7 slots ✅ MANAGEABLE

State Keeper Working Memory:
├─ State management: 6-7 slots (its specialty)
└─ No task logic (focused role)
```

**Delegation Pattern:**
```go
// Before: Agent manages state directly (complex)
func AgentDirect() {
  lock := AcquireLock("resource")  // 2 slots
  defer ReleaseLock(lock)          // Track in memory

  value := ReadState("counter")     // 1 slot
  WriteState("counter", value+1)    // 1 slot
  // Total: 4 slots just for state
}

// After: Agent delegates to State Keeper (simple)
func AgentDelegated() {
  Request(StateKeeper, IncrementCounter{Key: "counter"})
  // Total: 1 slot (just send request)
}
```

### Key Insight
**Delegation is cognitive compression.** Trade network cost for mental simplicity.

---

## Iteration 4: State Keeper Grammar - API

### Question
What operations should State Keeper expose?

### API Definition

```
STATE_KEEPER_API := {
  # Read Operations
  Get(key) → Option[Value]
  GetMulti(keys) → Map[Key, Option[Value]]

  # Write Operations
  Set(key, value) → Result[Unit]
  SetMulti(updates) → Result[Unit]
  Delete(key) → Result[Unit]

  # Atomic Operations
  Increment(key, delta) → Result[Int]
  CompareAndSwap(key, old, new) → Result[Bool]

  # Locks
  Lock(key, timeout) → Result[Guard]
  Unlock(guard) → Result[Unit]

  # Transactions
  Transaction(ops) → Result[T]
    where ops is sequence of state operations

  # Queries
  Keys(prefix) → []Key
  Size() → Int
  Exists(key) → Bool
}
```

**Transaction Example:**
```go
Transaction([
  Get("balance"),
  Set("balance", balance - amount),
  Append("ledger", transaction),
])
```

**Policies:**
```yaml
access_control:
  - agent: "worker-*"
    allow: ["Get", "Increment"]
    deny: ["Delete"]

  - agent: "admin"
    allow: ["*"]

rate_limit:
  - agent: "*"
    max_ops_per_second: 100

quota:
  - agent: "*"
    max_keys: 1000
    max_value_size: "10 MB"
```

### Key Insight
**API is request/reply over message protocol.** State Keeper is a specialized service agent.

---

## Iteration 5: Temporal Dynamics - Consistency vs Availability

### Question
How does State Keeper handle concurrent requests?

### Concurrency Analysis

**Consistency Model: Strong (Linearizable)**

```
State Keeper ensures linearizability:
- All operations appear to occur atomically at a single point in time
- Operations have total order
- Real-time constraints: if op1 completes before op2 starts, op1 < op2 in order
```

**Concurrency Control:**

```
Sequential (CP - Consistency + Partition Tolerance):
├─ Process one request at a time
├─ Pros: Simple, strong consistency
├─ Cons: Low throughput
└─ Use for: Critical state (locks, leader election)

Read-Write Locks (Hybrid):
├─ Multiple concurrent reads, exclusive writes
├─ Pros: Better throughput
├─ Cons: Starvation possible
└─ Use for: Read-heavy workloads

Optimistic (AP - Availability + Partition Tolerance):
├─ Process all requests concurrently
├─ Use CAS to detect conflicts
├─ Retry on conflict
├─ Pros: High throughput
├─ Cons: Retries on contention
└─ Use for: Low-contention state
```

**Tradeoff:**
```
Throughput ↔ Consistency

Sequential: Strong consistency, low throughput
Optimistic: Eventual consistency, high throughput

State Keeper chooses: Strong consistency (sequential)
Reasoning: Coordination state needs correctness > speed
```

### Key Insight
**State Keeper prioritizes consistency over throughput.** Use for coordination, not high-frequency operations.

---

## Iteration 6: Failure Modes - Resilience

### Question
What happens if State Keeper fails?

### Failure Analysis

**Single Point of Failure:**
- If State Keeper crashes, coordination state is unavailable
- All agents waiting on locks/transactions blocked

**Mitigation Strategies:**

**1. Persistence**
```yaml
strategy: "Write-Ahead Log (WAL)"
implementation:
  - Every state operation logged to durable storage
  - On restart, replay WAL to reconstruct state
recovery_time: "< 5 seconds"
```

**2. Replication**
```yaml
strategy: "Leader + Followers (Raft/Paxos)"
implementation:
  - Primary State Keeper handles writes
  - Followers replicate state
  - On leader failure, elect new leader
failover_time: "< 10 seconds"
complexity: "High (distributed consensus)"
```

**3. Timeouts**
```yaml
strategy: "Client-side timeouts"
implementation:
  - All requests have timeout (5s default)
  - If State Keeper doesn't respond, return error
  - Caller can retry or use fallback
graceful_degradation: true
```

**4. Circuit Breaker**
```yaml
strategy: "Stop calling if State Keeper is down"
implementation:
  - After N failures, open circuit
  - Don't send requests for cooldown period
  - Periodically probe to check recovery
prevents: "Thundering herd on restart"
```

**Recommended Configuration:**
```
Persistence: Always (low cost, high value)
Replication: For production only (adds complexity)
Timeouts: Always (required for liveness)
Circuit Breaker: Optional (nice-to-have)
```

### Key Insight
**Persistence + Timeouts provide baseline resilience.** Replication adds availability at cost of complexity.

---

## Iteration 7: Final Synthesis - Optimal State Keeper

### Synthesis

**STATE_KEEPER_AGENT** is an **L4_ADAPTIVE** agent that centralizes coordination state management.

**Core Architecture:**

```
┌───────────────────────────────────────┐
│        STATE KEEPER AGENT             │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │  Request Handler                │ │
│  │  (Message Protocol)             │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Policy Enforcement             │ │
│  │  (Access Control, Rate Limit)   │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Concurrency Control            │ │
│  │  (Locks, Transactions)          │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  State Store                    │ │
│  │  (Map[Key, Value])              │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Persistence (WAL)              │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
```

**Quality Metrics:**
```yaml
consistency: 1.0      # Linearizable (strong)
availability: 0.99    # 99% uptime (with persistence)
latency_p50: 5ms      # Median request latency
latency_p99: 20ms     # Tail latency
throughput: 1000 ops/s  # Sequential processing
```

**When to Use:**
```
✅ Use State Keeper for:
  - Distributed locks
  - Leader election
  - Shared counters/flags
  - Workflow coordination
  - Cross-agent state

❌ Don't use State Keeper for:
  - High-frequency operations (>1000 ops/s)
  - Agent-local state (use L2 local state)
  - Immutable config (use L1 context)
  - Append-only logs (use L3 artifacts)
```

### Key Insight
**State Keeper is a service agent** that provides strongly-consistent coordination state as a service to other agents.

---

## Meta-Learning

What did we learn about **State Keeper** through meta-prompting?

1. **Centralization reduces complexity** - One owner for coordination state
2. **Delegation reduces cognitive load** - Agents request vs manage
3. **Strong consistency over throughput** - Correctness matters for coordination
4. **API is request/reply** - State Keeper is a service
5. **Persistence is essential** - Survive failures and restarts
6. **Use sparingly** - Only for coordination, not general storage
7. **Single responsibility** - Manages Layer 4 only, nothing else

The pattern: **Centralize coordination state → Expose request/reply API → Enforce policies → Persist to WAL → Strong consistency guarantees**
