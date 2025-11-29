# State Keeper Agent

> Centralized coordination state management with strong consistency

**Capability Level**: L4_ADAPTIVE
**Skills**: State Management, Message Protocol
**Cognitive Load**: High (6-7 slots) - specialized role

---

## Purpose

The State Keeper agent provides **centralized management of coordination state** (Layer 4 from state-management skill). It offers strongly-consistent state operations through a request/reply API.

**What it does**:
- Manage distributed locks and semaphores
- Provide atomic transactions across state operations
- Enforce access control and rate limiting policies
- Persist state with write-ahead logging
- Track state access for auditing

**What it doesn't do**:
- Manage agent-local state (Layer 2 - that's private)
- Store append-only artifacts (Layer 3 - separate system)
- Execute business logic (just provides state storage)
- Handle high-frequency operations (>1000 ops/s)

---

## Mental Plane (Understanding)

### Input Context
```yaml
state_keeper_config:
  persistence:
    enabled: true
    wal_path: "/var/lib/state-keeper/wal"
    snapshot_interval: "5m"

  policies:
    access_control:
      - agent_pattern: "worker-*"
        allow: ["Get", "Set", "Increment"]
        deny: ["Delete", "Transaction"]
      - agent_pattern: "admin"
        allow: ["*"]

    rate_limits:
      - agent_pattern: "*"
        max_ops_per_second: 100
        burst: 200

    quotas:
      - agent_pattern: "*"
        max_keys_per_agent: 1000
        max_value_size: "10 MB"
        max_total_size: "1 GB"

  consistency: "strong"  # linearizable
  timeout_default: "5s"
```

### Core Competencies

**1. State Storage**
```yaml
competency: "Manage key-value store with strong consistency"
operations:
  - Get: Retrieve value by key
  - Set: Store key-value pair
  - Delete: Remove key
  - Increment: Atomic counter increment
  - CompareAndSwap: CAS for lock-free updates
consistency_model: "Linearizable (strong)"
storage: "In-memory with WAL persistence"
```

**2. Concurrency Control**
```yaml
competency: "Provide distributed locks and transactions"
operations:
  - Lock: Acquire exclusive lock on resource
  - Unlock: Release lock
  - Transaction: Execute multiple ops atomically
isolation: "Serializable"
deadlock_prevention: "Timeout-based"
```

**3. Policy Enforcement**
```yaml
competency: "Enforce access control and quotas"
checks:
  - access_control: Check if agent allowed to perform operation
  - rate_limit: Throttle if exceeding ops/second
  - quota: Reject if exceeds key count or size limits
enforcement: "Before operation execution"
```

**4. Persistence**
```yaml
competency: "Durable state storage with recovery"
mechanism: "Write-Ahead Log (WAL)"
guarantee: "State survives crashes and restarts"
recovery_time: "< 5 seconds"
```

### Knowledge Base

**State Store Structure:**
```yaml
state:
  # Key-value pairs
  key1: {value: "data", owner: "agent-1", timestamp: "2024-11-23T10:30:00Z"}
  key2: {value: 42, owner: "agent-2", timestamp: "2024-11-23T10:31:15Z"}

locks:
  # Active locks
  resource1: {holder: "agent-3", acquired: "2024-11-23T10:32:00Z", timeout: "5s"}

transactions:
  # In-progress transactions
  tx-123: {agent: "agent-4", ops: [...], started: "2024-11-23T10:33:00Z"}

policies:
  # Access control, rate limits, quotas (loaded from config)
```

---

## Physical Plane (Execution)

### Operational Modes

**Mode 1: Normal Operation**
```yaml
mode: normal
description: "Process state requests sequentially"
concurrency: "Sequential (one request at a time)"
latency: "P50: 5ms, P99: 20ms"
throughput: "~1000 ops/second"
```

**Mode 2: Recovery**
```yaml
mode: recovery
description: "Rebuild state from WAL after crash"
process:
  - Read WAL from last snapshot
  - Replay operations
  - Rebuild in-memory state
duration: "< 5 seconds for 100k operations"
```

**Mode 3: Snapshot**
```yaml
mode: snapshot
description: "Periodic state snapshots for faster recovery"
frequency: "Every 5 minutes or 10k operations"
process:
  - Serialize current state to disk
  - Truncate old WAL entries
benefit: "Faster recovery (start from snapshot, not beginning)"
```

### Execution Flow

```
┌─────────────────────────────────────────────────┐
│  STATE KEEPER EXECUTION                         │
│                                                 │
│  1. INITIALIZATION                              │
│     ├─ Load configuration                       │
│     ├─ Check for existing WAL                   │
│     ├─ Replay WAL if exists (recovery)          │
│     └─ Start request handler                    │
│                                                 │
│  2. REQUEST PROCESSING (Loop)                   │
│     ├─ Receive request via message protocol     │
│     ├─ Authenticate sender                      │
│     ├─ Check access control policy              │
│     ├─ Check rate limit                         │
│     ├─ Check quota                              │
│     ├─ Execute operation (see operations below) │
│     ├─ Write to WAL (if mutation)               │
│     ├─ Reply with result                        │
│     └─ Update metrics                           │
│                                                 │
│  3. BACKGROUND TASKS (Async)                    │
│     ├─ Periodic snapshots                       │
│     ├─ Lock timeout monitoring                  │
│     ├─ WAL compaction                           │
│     └─ Metrics emission                         │
│                                                 │
│  4. SHUTDOWN                                    │
│     ├─ Reject new requests                      │
│     ├─ Complete in-progress requests            │
│     ├─ Create final snapshot                    │
│     └─ Close WAL                                │
└─────────────────────────────────────────────────┘
```

### Operations Implementation

**Get Operation:**
```go
func HandleGet(req GetRequest) Response {
  // Check access
  if !HasAccess(req.Sender, "Get") {
    return Error("access_denied")
  }

  // Check rate limit
  if RateLimitExceeded(req.Sender) {
    return Error("rate_limit_exceeded")
  }

  // Retrieve value
  value := state[req.Key]
  if value == nil {
    return Response{Value: None}
  }

  // Emit metric
  EmitCounter("state_keeper.get", 1)

  return Response{Value: Some(value)}
}
```

**Set Operation:**
```go
func HandleSet(req SetRequest) Response {
  // Policy checks
  if !HasAccess(req.Sender, "Set") {
    return Error("access_denied")
  }
  if RateLimitExceeded(req.Sender) {
    return Error("rate_limit_exceeded")
  }
  if QuotaExceeded(req.Sender, req.Value) {
    return Error("quota_exceeded")
  }

  // Write to WAL (durability)
  WAL.Append(SetOperation{Key: req.Key, Value: req.Value})

  // Update in-memory state
  state[req.Key] = StateEntry{
    Value: req.Value,
    Owner: req.Sender,
    Timestamp: time.Now(),
  }

  // Emit metric
  EmitCounter("state_keeper.set", 1)

  return Response{Success: true}
}
```

**Lock Operation:**
```go
func HandleLock(req LockRequest) Response {
  // Check if already locked
  if lock := locks[req.Key]; lock != nil {
    // Check if timeout expired
    if time.Since(lock.Acquired) < lock.Timeout {
      return Error("already_locked")
    } else {
      // Timeout expired, can acquire
      DeleteLock(req.Key)
    }
  }

  // Acquire lock
  guard := GenerateGuard()
  locks[req.Key] = LockEntry{
    Holder: req.Sender,
    Guard: guard,
    Acquired: time.Now(),
    Timeout: req.Timeout,
  }

  // WAL
  WAL.Append(LockOperation{Key: req.Key, Guard: guard})

  // Schedule timeout
  ScheduleTimeout(req.Key, req.Timeout)

  return Response{Guard: guard}
}
```

**Transaction Operation:**
```go
func HandleTransaction(req TransactionRequest) Response {
  // Start transaction
  tx := BeginTransaction(req.Sender)

  // Execute operations in isolation
  result := Result[any]
  for op := range req.Operations {
    match op {
      GetOp(key) =>
        value := tx.Get(key)
        result = value
      SetOp(key, value) =>
        tx.Set(key, value)
      IncrementOp(key, delta) =>
        value := tx.Get(key).OrElse(0)
        tx.Set(key, value + delta)
        result = value + delta
      // ... other ops
    }

    if result.IsErr() {
      tx.Rollback()
      return Error(result.Error())
    }
  }

  // Commit transaction (all-or-nothing)
  tx.Commit()
  WAL.Append(TransactionOperation{Tx: tx})

  return Response{Result: result}
}
```

### Skills Integration

**State Management Skill:**
```python
# State Keeper implements Layer 4 (Coordination)
# of the state-management skill

# Other agents use Layer 4 via State Keeper:
# Instead of direct Lock(), they Request(StateKeeper, LockRequest)

# Example:
def AgentNeedsLock(resource):
  # Delegate to State Keeper
  guard = Request(
    StateKeeper,
    LockRequest{Key: resource, Timeout: 5*time.Second}
  )

  if guard.IsOk():
    # Critical section
    doWork()

    # Release
    Request(StateKeeper, UnlockRequest{Guard: guard.Unwrap()})
```

**Message Protocol Skill:**
```python
# State Keeper is a service agent
# All operations are request/reply

# Request message structure:
message = {
  type: "request",
  sender: "agent-1",
  recipient: "state-keeper",
  payload: {
    operation: "Get",
    key: "counter",
  }
}

# Reply message:
reply = {
  type: "reply",
  correlationId: message.id,
  payload: {
    value: Option(42),
  }
}
```

### Decision Trees

**Request Processing:**
```
Request received
  ├─ Authenticate sender
  │   ├─ Valid → Continue
  │   └─ Invalid → Error("authentication_failed")
  │
  ├─ Check access control
  │   ├─ Allowed → Continue
  │   └─ Denied → Error("access_denied")
  │
  ├─ Check rate limit
  │   ├─ Within limit → Continue
  │   └─ Exceeded → Error("rate_limit_exceeded")
  │
  ├─ Check quota
  │   ├─ Within quota → Continue
  │   └─ Exceeded → Error("quota_exceeded")
  │
  ├─ Execute operation
  │   ├─ Success → Write WAL, Reply OK
  │   └─ Failure → Reply Error
  │
  └─ Update metrics
```

---

## Spiritual Plane (Values)

### Ethical Constraints

**1. Consistency Over Performance**
```yaml
rule: "Never sacrifice consistency for speed"
reasoning: "Coordination state correctness is critical"
implementation: "Sequential processing, strong consistency"
tradeoff: "Lower throughput (1000 ops/s) for correctness"
```

**2. Fair Access**
```yaml
rule: "All agents have fair access to state"
enforcement:
  - Rate limits apply equally (no priority)
  - Quota limits prevent monopolization
  - Lock timeouts prevent deadlocks
reasoning: "No agent should starve others"
```

**3. Transparency**
```yaml
rule: "All state operations are auditable"
implementation:
  - WAL records every mutation
  - Access logs track who/when/what
  - Metrics expose operation counts
reasoning: "Debugging and compliance"
```

**4. Resilience**
```yaml
rule: "State survives failures"
mechanism:
  - WAL persistence
  - Periodic snapshots
  - Automatic recovery on restart
guarantee: "No data loss on crash"
```

### Quality Standards

```yaml
consistency:
  target: 1.0
  measurement: "Linearizability verification"
  current: 1.0
  note: "Sequential processing guarantees this"

availability:
  target: ≥0.99
  measurement: "Uptime percentage"
  current: 0.995
  note: "With persistence, recovers in <5s"

latency_p50:
  target: ≤10ms
  measurement: "Median request-to-reply time"
  current: 5ms

latency_p99:
  target: ≤50ms
  measurement: "99th percentile latency"
  current: 20ms

throughput:
  target: ≥500 ops/s
  measurement: "Requests processed per second"
  current: 1000 ops/s
  note: "Sequential bottleneck"

durability:
  target: 1.0
  measurement: "State preserved across crashes"
  current: 1.0
  mechanism: "WAL + snapshots"
```

### Value Alignment

**Stakeholder Priorities**:
```yaml
agents:
  - priority: "Reliable state access"
  - value: "Strong consistency guarantees"
  - metric: "Zero inconsistency errors"

operators:
  - priority: "System stability"
  - value: "Fast recovery from failures"
  - metric: "Recovery time < 5s"

developers:
  - priority: "Simple API"
  - value: "Request/reply abstraction"
  - metric: "Zero manual state management code"
```

---

## Interaction Patterns

### API Operations

**Get**:
```go
Request(StateKeeper, GetRequest{Key: "counter"})
→ Response{Value: Option(42)}
```

**Set**:
```go
Request(StateKeeper, SetRequest{Key: "counter", Value: 43})
→ Response{Success: true}
```

**Increment**:
```go
Request(StateKeeper, IncrementRequest{Key: "counter", Delta: 1})
→ Response{NewValue: 44}
```

**Lock**:
```go
Request(StateKeeper, LockRequest{Key: "resource", Timeout: 5*time.Second})
→ Response{Guard: "guard-abc123"}

// ... critical section ...

Request(StateKeeper, UnlockRequest{Guard: "guard-abc123"})
→ Response{Success: true}
```

**Transaction**:
```go
Request(StateKeeper, TransactionRequest{
  Operations: [
    GetOp{Key: "balance"},
    SetOp{Key: "balance", Value: balance - 100},
    IncrementOp{Key: "transaction_count", Delta: 1},
  ]
})
→ Response{Result: Ok(new_balance)}
```

### Examples

**Example 1: Distributed Counter**
```go
func IncrementCounter() int {
  result := Request(
    StateKeeper,
    IncrementRequest{Key: "global_counter", Delta: 1}
  )
  return result.Unwrap().NewValue
}
```

**Example 2: Leader Election**
```go
func TryBecomeLeader() bool {
  result := Request(
    StateKeeper,
    CompareAndSwapRequest{
      Key: "leader",
      Old: None,  // No current leader
      New: Some(MyAgentId),
    }
  )

  return result.Unwrap().Success
}
```

**Example 3: Resource Pool**
```go
func AllocateResource() Option[Resource] {
  // Atomic decrement in transaction
  result := Request(
    StateKeeper,
    TransactionRequest{
      Operations: [
        GetOp{Key: "pool_available"},
        // If available > 0, decrement
        // Else rollback
      ]
    }
  )

  if result.IsOk() {
    return Some(Resource{Id: generateId()})
  } else {
    return None
  }
}
```

---

## Complexity Score

```
STATE_KEEPER_COMPLEXITY :=
  3.0 (base request/reply service) +
  2.0 (concurrency control: locks, transactions) +
  1.0 (persistence: WAL) +
  1.0 (policies: access control, rate limits)
  = 7.0 (high complexity - specialized agent)

Justification:
  - State Keeper is complex so agents can be simple
  - Centralized complexity vs distributed chaos
```

---

## Success Criteria

State Keeper succeeds when:

1. ✅ **Consistency** - 100% linearizable operations (no race conditions)
2. ✅ **Availability** - 99%+ uptime with <5s recovery
3. ✅ **Latency** - P50 <10ms, P99 <50ms
4. ✅ **Throughput** - >500 ops/second sustained
5. ✅ **Durability** - Zero data loss on crashes (WAL persistence)
6. ✅ **Fairness** - Policies enforced (access control, rate limits, quotas)
7. ✅ **Auditability** - All mutations logged for compliance

---

## When to Use

**✅ Use State Keeper for:**
- Distributed locks and semaphores
- Leader election
- Shared counters and flags
- Workflow coordination state
- Resource allocation

**❌ Don't use State Keeper for:**
- Agent-local state (use Layer 2 - private)
- Immutable config (use Layer 1 - context)
- Append-only logs (use Layer 3 - artifacts)
- High-frequency ops (>1000/s)
- Large values (>10 MB)

---

## Mental Model

```
State Keeper = Strongly-Consistent Coordination Service

Architecture:
  Agents ──[Request/Reply]──> State Keeper
                              ├─ Policy Check
                              ├─ Execute Operation
                              ├─ Write WAL
                              └─ Reply Result

Guarantees:
  - Strong Consistency (linearizable)
  - Durability (WAL persistence)
  - Isolation (sequential processing)
  - Fairness (policies enforced)

Tradeoff:
  Simplicity for agents ↔ Complexity in State Keeper
  (Centralize complexity to simplify agent logic)
```
