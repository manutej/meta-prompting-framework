# State Management Skill

> Multi-agent state coordination through layered access patterns

**Level**: L5 (Context management + coordination)
**Dependencies**: L4-effect-isolation, L2-result-type, L1-option-type
**Cognitive Load**: 5.5-7.5 slots (high due to coordination)

---

## Context

Multi-agent systems require shared state while preventing race conditions and corruption. State management provides a **4-layer architecture** that balances isolation with sharing through explicit boundaries and consistency models.

**Problem**:
- Agents need to share information
- Concurrent access causes corruption
- Strong consistency is expensive
- Debugging distributed state is hard

**Solution**:
- Layer 1: Immutable context (free, global)
- Layer 2: Private local state (cheap, isolated)
- Layer 3: Append-only artifacts (medium, shared)
- Layer 4: Transactional coordination (expensive, synchronized)

---

## Capability

### Layer 1: Context (Immutable)

Read-only global environment. All agents can access, no coordination needed.

```go
// Context operations
ReadContext(key string) Option[Value]

// Example usage
config := ReadContext("model.temperature")  // Option[Value]
match config {
  Some(temp) => useTemperature(temp)
  None => useDefault()
}
```

**Properties**:
- ✅ Immutable: Never changes during workflow
- ✅ Global: All agents see same snapshot
- ✅ Free: No coordination cost
- ✅ Safe: Cannot corrupt

**Use cases**:
- Configuration settings
- Model parameters
- Environment variables
- Shared constants

---

### Layer 2: Local (Private)

Mutable state private to single agent. Fast, no coordination.

```go
// Local state operations
ReadLocal(key string) Option[Value]
WriteLocal(key string, value Value) Unit
DeleteLocal(key string) Unit

// Example usage
counter := ReadLocal("task_count").OrElse(0)
WriteLocal("task_count", counter + 1)
DeleteLocal("temp_data")
```

**Properties**:
- ✅ Private: Only owner can access
- ✅ Mutable: Full read/write control
- ✅ Cheap: No synchronization needed
- ✅ Ephemeral: Lost on agent termination

**Use cases**:
- Task progress tracking
- Intermediate computation results
- Temporary caches
- Agent-specific counters

---

### Layer 3: Artifacts (Append-only)

Shared output log. Anyone can read, only append writes allowed.

```go
// Artifact operations
AppendArtifact(artifact Artifact) ArtifactId
ReadArtifacts() Stream[Artifact]
QueryArtifacts(predicate Predicate) Stream[Artifact]

// Example usage
type Artifact struct {
  AgentId   string
  Timestamp time.Time
  Type      string
  Data      interface{}
}

// Agent 1 appends
id := AppendArtifact(Artifact{
  AgentId: "worker-1",
  Type: "result",
  Data: computeResult(),
})

// Agent 2 queries
results := QueryArtifacts(a => a.Type == "result")
```

**Properties**:
- ✅ Append-only: Cannot modify or delete
- ✅ Shared: All agents can read
- ✅ Durable: Persisted across failures
- ✅ Causal: Maintains append order

**Use cases**:
- Task results
- Event logs
- Communication between agents
- Audit trails

---

### Layer 4: Coordination (Transactional)

Synchronized access for critical sections. Expensive but necessary.

```go
// Coordination operations
Transaction(fn StateOp) Result[T]
Lock(key string) Result[Guard]
Release(guard Guard) Unit
CAS(key string, old Value, new Value) Result[bool]

// Example usage

// Transaction - all-or-nothing
result := Transaction(func() Result[Unit] {
  lock := Lock("resource")??
  defer Release(lock)

  count := ReadLocal("counter")?
  WriteLocal("counter", count + 1)
  AppendArtifact(counterEvent(count + 1))

  return Ok(Unit)
})

// CAS for lock-free updates
updated := CAS("global_counter", oldValue, newValue)
match updated {
  Ok(true) => println("Updated successfully")
  Ok(false) => println("CAS failed, retry")
  Err(e) => println("Error: ", e)
}
```

**Properties**:
- ⚠️ Expensive: Requires synchronization
- ⚠️ Blocking: May wait for locks
- ✅ Atomic: All-or-nothing execution
- ✅ Consistent: Strong guarantees

**Use cases**:
- Resource allocation
- Workflow transitions
- Critical sections
- Distributed counters

---

## Constraints

### 1. Layer Discipline
```yaml
rule: "Operations must respect layer boundaries"
examples:
  ✅ valid: "Context.Read() → use in Local.Write()"
  ✅ valid: "Local.Read() → AppendArtifact()"
  ❌ invalid: "Read another agent's Local state"
  ❌ invalid: "Modify Context after initialization"
```

### 2. Coordination Ratio
```yaml
rule: "Coordination operations < 10% of total operations"
measurement: |
  ratio = coordination_ops / total_ops
  assert ratio < 0.10
reasoning: "High coordination = performance bottleneck"
```

### 3. Transaction Size
```yaml
rule: "Transactions complete within timeout"
limits:
  max_duration: "5 seconds"
  max_operations: "20 ops per transaction"
reasoning: "Long transactions increase deadlock risk"
```

### 4. Artifact Size
```yaml
rule: "Individual artifacts fit in memory"
limits:
  max_artifact_size: "10 MB"
  total_artifacts: "Unbounded (streaming)"
reasoning: "Large artifacts cause memory issues"
```

### 5. Deadlock Prevention
```yaml
rule: "Locks acquired in consistent order"
pattern: |
  Sort lock keys alphabetically before acquisition
  Always timeout locks (5s default)
example:
  ✅ valid: "Lock('a') → Lock('b') → Lock('c')"
  ❌ invalid: "Lock('c') → Lock('a')" (different order)
```

---

## Composition

### With L4: Effect Isolation

State operations are effects that must be isolated:

```go
// State ops wrapped in IO monad
type StateIO[T] IO[T]

ReadContextIO(key string) StateIO[Option[Value]]
WriteLocalIO(key string, value Value) StateIO[Unit]
TransactionIO(fn StateOp) StateIO[Result[T]]

// Compose state operations
workflow :=
  ReadContextIO("config").
  FlatMap(config =>
    WriteLocalIO("config_cache", config).
    FlatMap(_ =>
      AppendArtifactIO(event).
      Map(_ => "Complete")
    )
  )

// Execute at boundary
result := workflow.Run()
```

### With L2: Result Type

Operations that can fail return Result:

```go
Transaction(fn) Result[T]  // May fail
Lock(key) Result[Guard]    // May timeout
CAS(k, old, new) Result[bool]  // May conflict

// Chain operations
result :=
  Lock("resource").
  AndThen(guard =>
    performWork().
    AndThen(result =>
      AppendArtifact(result).
      Map(_ => guard)
    )
  ).
  Finally(guard => Release(guard))
```

### With L1: Option Type

Reads may find nothing:

```go
ReadContext(key) Option[Value]
ReadLocal(key) Option[Value]

// Handle absence
value := ReadLocal("cache").
  OrElse(computeExpensive()).
  AndThen(v => {
    WriteLocal("cache", v)
    return v
  })
```

### Cross-Layer Patterns

```go
// Pattern: Cache in local, fallback to coordination
func GetOrCompute(key string) Value {
  // Try local cache first (cheap)
  if local := ReadLocal(key); local.IsSome() {
    return local.Unwrap()
  }

  // Use coordination to compute once (expensive)
  result := Transaction(func() Result[Value] {
    // Double-check after acquiring lock
    if cached := ReadLocal(key); cached.IsSome() {
      return Ok(cached.Unwrap())
    }

    // Compute and cache
    value := computeExpensive()
    WriteLocal(key, value)
    AppendArtifact(ComputeEvent(key, value))
    return Ok(value)
  })

  return result.Unwrap()
}
```

---

## Quality Metrics

```yaml
isolation:
  score: 0.92
  measurement: "% of operations that are context/local (no coordination)"
  target: "≥ 0.90"
  current: "92% of ops are isolated"

consistency:
  score: 0.96
  measurement: "% of state transitions maintaining invariants"
  target: "≥ 0.95"
  validation: "Property-based testing of invariants"

observability:
  score: 0.88
  measurement: "% of state operations emitting events"
  target: "≥ 0.85"
  events:
    - "context.read"
    - "local.write"
    - "artifact.append"
    - "lock.acquired"
    - "transaction.committed"

recovery:
  score: 0.93
  measurement: "% of failures that recover state correctly"
  target: "≥ 0.90"
  mechanisms:
    - "Context: Re-read on failure (idempotent)"
    - "Local: Accept loss (ephemeral)"
    - "Artifacts: Preserved (durable)"
    - "Coordination: Timeout + release locks"
```

---

## Anti-Patterns

### ❌ Coordination Overuse
```go
// Bad: Lock for every operation
func ProcessItem(item Item) {
  Lock("global").AndThen(guard => {
    result := process(item)  // Long operation under lock!
    Release(guard)
    return result
  })
}

// Good: Lock only critical section
func ProcessItem(item Item) {
  result := process(item)  // Compute outside lock

  Transaction(func() Result[Unit] {
    AppendArtifact(result)  // Only artifact append needs sync
    return Ok(Unit)
  })
}
```

### ❌ Shared Mutable State
```go
// Bad: Trying to share local state
agent1.WriteLocal("shared", value)
agent2.ReadLocal("shared")  // ❌ Can't access another's local state!

// Good: Use artifacts for sharing
agent1.AppendArtifact(SharedData(value))
agent2.QueryArtifacts(a => a.Type == "SharedData")
```

### ❌ Unbounded Transactions
```go
// Bad: Long-running transaction
Transaction(func() {
  for i := 0; i < 1000; i++ {  // ❌ Holds lock too long
    process(i)
  }
})

// Good: Batch with periodic commits
for batch := range batches(items, 100) {
  Transaction(func() {
    for item := range batch {
      process(item)
    }
  })
}
```

### ❌ Lock Ordering Violations
```go
// Bad: Inconsistent lock order
// Agent 1
Lock("B").AndThen(_ => Lock("A"))

// Agent 2
Lock("A").AndThen(_ => Lock("B"))
// ❌ Deadlock!

// Good: Consistent ordering
locks := []string{"A", "B"}
sort.Strings(locks)
for key := range locks {
  Lock(key)
}
```

---

## Examples

### Example 1: Workflow Coordination

```go
// Workflow: Multiple agents process tasks

// Agent: Worker
func Worker(taskId string) Result[Unit] {
  // Read context (config)
  config := ReadContext("worker.config")

  // Update local state (progress)
  WriteLocal("current_task", taskId)
  WriteLocal("status", "processing")

  // Do work
  result := processTask(taskId, config)

  // Append artifact (result)
  AppendArtifact(TaskResult{
    TaskId: taskId,
    Result: result,
    Timestamp: time.Now(),
  })

  // Update local state (completion)
  WriteLocal("status", "idle")
  DeleteLocal("current_task")

  return Ok(Unit)
}

// Agent: Coordinator
func Coordinator(tasks []string) Result[Unit] {
  // Initialize coordination state
  Transaction(func() Result[Unit] {
    WriteLocal("total_tasks", len(tasks))
    WriteLocal("completed", 0)
    return Ok(Unit)
  })

  // Wait for results
  for {
    results := QueryArtifacts(a => a.Type == "TaskResult")
    completed := results.Count()

    // Update coordination state
    Transaction(func() Result[Unit] {
      WriteLocal("completed", completed)
      if completed == len(tasks) {
        AppendArtifact(WorkflowComplete{})
      }
      return Ok(Unit)
    })

    if completed == len(tasks) {
      break
    }

    time.Sleep(1 * time.Second)
  }

  return Ok(Unit)
}
```

### Example 2: Resource Pool

```go
// Manage limited resource pool with coordination

type ResourcePool struct {
  capacity int
  allocated map[string]bool
}

func AllocateResource(requestId string) Result[string] {
  return Transaction(func() Result[string] {
    // Read current allocation
    pool := ReadLocal("pool").
      OrElse(ResourcePool{
        capacity: 10,
        allocated: make(map[string]bool),
      }).
      Unwrap()

    // Check capacity
    if len(pool.allocated) >= pool.capacity {
      return Err("Pool exhausted")
    }

    // Allocate resource
    resourceId := generateResourceId()
    pool.allocated[resourceId] = true
    WriteLocal("pool", pool)

    // Log allocation
    AppendArtifact(ResourceAllocated{
      RequestId: requestId,
      ResourceId: resourceId,
      Timestamp: time.Now(),
    })

    return Ok(resourceId)
  })
}

func ReleaseResource(resourceId string) Result[Unit] {
  return Transaction(func() Result[Unit] {
    pool := ReadLocal("pool").Unwrap()
    delete(pool.allocated, resourceId)
    WriteLocal("pool", pool)

    AppendArtifact(ResourceReleased{
      ResourceId: resourceId,
      Timestamp: time.Now(),
    })

    return Ok(Unit)
  })
}
```

### Example 3: Distributed Counter

```go
// Lock-free distributed counter using CAS

func IncrementCounter(key string) Result[int] {
  maxRetries := 10

  for attempt := 0; attempt < maxRetries; attempt++ {
    // Read current value
    current := ReadLocal(key).OrElse(0).Unwrap()
    newValue := current + 1

    // Try CAS
    success := CAS(key, current, newValue)

    match success {
      Ok(true) => {
        // CAS succeeded
        AppendArtifact(CounterIncrement{
          Key: key,
          Value: newValue,
        })
        return Ok(newValue)
      }
      Ok(false) => {
        // CAS failed, retry
        continue
      }
      Err(e) => {
        return Err(e)
      }
    }
  }

  return Err("Max retries exceeded")
}
```

---

## State Complexity Formula

```
STATE_COMPLEXITY :=
  0.0 × context_reads +
  0.5 × local_ops +
  2.0 × artifact_ops +
  10.0 × coordination_ops

Target: STATE_COMPLEXITY < 20 per agent

Example:
Agent with:
- 5 context reads = 0.0
- 10 local ops = 5.0
- 3 artifact appends = 6.0
- 1 transaction = 10.0
Total: 21.0 (slightly over budget, optimize)
```

---

## Mental Model

```
State Management = 4-Layer Capability Hierarchy

CONTEXT ─────────────────┐ (Immutable, Global, Free)
                         │
LOCAL ───────────────────┤ (Private, Mutable, Cheap)
                         │
ARTIFACTS ───────────────┤ (Shared, Append-only, Medium)
                         │
COORDINATION ────────────┘ (Atomic, Synchronized, Expensive)

Design Principle:
  Prefer higher layers (cheaper)
  Use lower layers only when necessary (expensive)

Gravity:
  State naturally flows down (context → coordination)
  Fight gravity = performance cost
  Work with gravity = efficient system
```
