# Message Protocol Skill

> Reliable, typed agent-to-agent communication

**Level**: L4 (Effect management with communication)
**Dependencies**: L2-result-type, L1-option-type
**Cognitive Load**: 2-4 slots (depends on pattern)

---

## Context

Multi-agent systems require explicit communication. Message passing provides **traceable, reliable, and composable** communication through structured messages with well-defined semantics.

**Problem**:
- Agents need to coordinate without shared memory
- Communication should be traceable for debugging
- Failures (timeouts, loss, duplicates) must be handled
- Different patterns needed (fire-and-forget, request/reply, pub/sub)

**Solution**:
- Typed messages with metadata (sender, recipient, correlation)
- Multiple patterns (send, request/reply, publish/subscribe)
- Reliable delivery with retries and deduplication
- Dead letter queues for failed messages
- Observable messaging (all operations emit events)

---

## Capability

### Core Message Structure

```go
type Message struct {
  Id            MessageId      // Unique identifier
  Type          MessageType    // send | request | reply | publish
  Sender        AgentId        // Who sent it
  Recipient     Target         // Where it goes
  CorrelationId Option[MessageId]  // Links request/reply
  Timestamp     Time           // When sent
  Payload       Data           // Message content
  Metadata      Metadata       // Optional context
}

type Target =
  | AgentId      // Direct to specific agent
  | Topic        // Broadcast to topic subscribers
  | Broadcast    // All agents in scope
```

---

### Pattern 1: Send (Fire-and-Forget)

Asynchronous, no reply expected.

```go
// Send message
Send(recipient Target, payload Data) MessageId

// Example usage
msgId := Send(
  AgentId("worker-1"),
  TaskNotification{
    TaskId: "task-123",
    Status: "completed",
  }
)

// Sender continues immediately (non-blocking)
doNextThing()
```

**Properties**:
- ✅ Non-blocking (async)
- ✅ Low cognitive load (1 slot)
- ✅ Fast (no waiting)
- ⚠️ No confirmation
- ⚠️ May fail silently (use monitoring)

**Use cases**:
- Notifications
- Event broadcasting
- Status updates
- Logging

---

### Pattern 2: Request/Reply (Synchronous)

Caller blocks until reply received.

```go
// Send request, wait for reply
Request(recipient AgentId, payload Data, timeout Duration) Result[Data]

// Example usage
result := Request(
  AgentId("database-agent"),
  QueryRequest{Query: "SELECT * FROM tasks WHERE id = ?", Args: ["task-123"]},
  timeout=5*time.Second
)

match result {
  Ok(data) => {
    println("Got result:", data)
  }
  Err("timeout") => {
    println("Request timed out after 5s")
  }
  Err(e) => {
    println("Request failed:", e)
  }
}
```

**Replying**:
```go
// Handler receives request
func OnRequest(msg Message) {
  // Process request
  result := processQuery(msg.Payload)

  // Send reply (links via correlation ID)
  Reply(msg.Id, result)
}
```

**Properties**:
- ✅ Synchronous (get immediate answer)
- ✅ Correlation automatic
- ⚠️ Blocking (2-3 slots)
- ⚠️ Timeout required
- ⚠️ Can deadlock if circular

**Use cases**:
- Queries (get state)
- RPC calls
- Resource requests
- Synchronous commands

---

### Pattern 3: Publish/Subscribe (Multicast)

One-to-many event broadcasting.

```go
// Subscribe to topic
Subscribe(topic Topic, handler Fn[Data → Unit]) SubscriptionId

// Publish to topic
Publish(topic Topic, payload Data) Unit

// Unsubscribe
Unsubscribe(subscriptionId SubscriptionId) Unit

// Example usage

// Agent A subscribes
subId := Subscribe(
  Topic("task.completed"),
  func(data Data) {
    task := data.(TaskEvent)
    println("Task completed:", task.TaskId)
  }
)

// Agent B publishes
Publish(
  Topic("task.completed"),
  TaskEvent{
    TaskId: "task-123",
    Result: "success",
  }
)
// → Agent A's handler is called

// Later: unsubscribe
Unsubscribe(subId)
```

**Properties**:
- ✅ Decoupled (publisher doesn't know subscribers)
- ✅ Dynamic (subscribe/unsubscribe at runtime)
- ✅ Scalable (add subscribers without changing publisher)
- ⚠️ No delivery guarantee per subscriber
- ⚠️ Topic management overhead

**Use cases**:
- Event notifications
- Monitoring alerts
- State change broadcasts
- Audit logging

---

### Pattern 4: Broadcast

Send to all agents in scope.

```go
// Broadcast to all agents
Broadcast(scope Scope, payload Data) MessageId

type Scope =
  | All              // Every agent
  | AllWorkers       // All agents with role="worker"
  | AllInWorkflow    // All agents in current workflow

// Example usage
Broadcast(
  AllWorkers,
  ShutdownNotice{
    Reason: "System maintenance",
    Deadline: time.Now().Add(5*time.Minute),
  }
)
```

**Properties**:
- ✅ Simple (no targeting logic)
- ✅ Fast (parallel delivery)
- ⚠️ No order guarantee across recipients
- ⚠️ Can't confirm all received

**Use cases**:
- System announcements
- Shutdown signals
- Configuration updates

---

### Pattern 5: Pipeline

Sequential message transformation.

```go
// Create pipeline
type Pipeline = []AgentId

SendToPipeline(pipeline Pipeline, payload Data) MessageId

// Example usage
pipeline := Pipeline{
  AgentId("validator"),
  AgentId("enricher"),
  AgentId("processor"),
  AgentId("storage"),
}

SendToPipeline(pipeline, RawData{...})

// Each stage:
// 1. Receives message
// 2. Transforms payload
// 3. Forwards to next stage
```

**Stage Implementation**:
```go
func PipelineStage(msg Message, nextStage Option[AgentId]) {
  // Transform
  transformed := transform(msg.Payload)

  // Forward
  match nextStage {
    Some(next) => Send(next, transformed)
    None => finalize(transformed)  // Last stage
  }
}
```

**Properties**:
- ✅ Clear flow (A → B → C)
- ✅ Composable stages
- ⚠️ Sequential (no parallelism)
- ⚠️ Backpressure from slow stages

**Use cases**:
- Data transformation pipelines
- Multi-stage validation
- Processing workflows

---

### Pattern 6: Scatter-Gather

Parallel fan-out, then collect.

```go
// Scatter to multiple agents, gather results
ScatterGather(
  recipients []AgentId,
  payload Data,
  timeout Duration
) Result[[]Data]

// Example usage
results := ScatterGather(
  []AgentId{"worker-1", "worker-2", "worker-3"},
  SearchQuery{Term: "error"},
  timeout=10*time.Second
)

match results {
  Ok(responses) => {
    // Got all responses
    combined := aggregate(responses)
  }
  Err("partial_timeout") => {
    // Some agents didn't respond in time
    // Partial results available
  }
  Err(e) => {
    // Complete failure
  }
}
```

**Properties**:
- ✅ Parallel execution
- ✅ Aggregate results
- ⚠️ High cognitive load (4-6 slots)
- ⚠️ Complex timeout logic
- ⚠️ Partial failure handling

**Use cases**:
- Parallel search
- Map-reduce
- Voting/consensus
- Load distribution

---

### Mailbox Operations

```go
// Receive next message (blocking)
Receive() Option[Message]

// Receive first matching message
ReceiveWhere(predicate Fn[Message → Bool]) Option[Message]

// Peek at next message without removing
Peek() Option[Message]

// Check mailbox size
MailboxSize() Int

// Example usage

// Receive any message
msg := Receive()
match msg {
  Some(m) => process(m)
  None => println("Mailbox empty")
}

// Selective receive (only errors)
errorMsg := ReceiveWhere(m => m.Type == "error")

// Check for backpressure
if MailboxSize() > 900 {
  emitBackpressure()
}
```

---

### Reliability Operations

```go
// Acknowledge successful processing
Ack(messageId MessageId) Unit

// Negative acknowledge (processing failed)
Nack(messageId MessageId, reason Reason) Unit

// Move to dead letter queue
DeadLetter(message Message, reason Reason) Unit

// Example usage
msg := Receive().Unwrap()

result := processMessage(msg)
match result {
  Ok(_) => Ack(msg.Id)
  Err(e) => {
    if retriable(e) {
      Nack(msg.Id, e)  // Will retry
    } else {
      DeadLetter(msg, e)  // Permanent failure
    }
  }
}
```

---

## Constraints

### 1. Mailbox Capacity
```yaml
rule: "Mailboxes bounded to prevent memory exhaustion"
capacity: 1000
backpressure_threshold: 900  # 90%
overflow_policy: "drop_oldest" | "block_sender"
```

### 2. Message Ordering
```yaml
rule: "FIFO per sender-receiver pair"
guarantee: |
  Send(A, B, m1); Send(A, B, m2)
  => B receives m1 before m2

no_guarantee: |
  Send(A, C, m1); Send(B, C, m2)
  => C may receive in any order
```

### 3. Timeout Bounds
```yaml
rule: "All blocking operations have timeout"
request_timeout: "≤ 30s (default 5s)"
scatter_gather_timeout: "≤ 60s"
reasoning: "Prevent indefinite blocking"
```

### 4. Deduplication Window
```yaml
rule: "Track seen message IDs to detect duplicates"
window: "Last 10,000 message IDs"
retention: "1 hour"
reasoning: "Balance memory vs duplicate detection"
```

### 5. Dead Letter Retention
```yaml
rule: "Failed messages preserved for debugging"
retention: "7 days"
max_size: "10,000 messages"
reasoning: "Investigate failures without losing evidence"
```

---

## Composition

### With L2: Result Type

Operations that can fail return Result:

```go
Request(recipient, payload, timeout) Result[Data]
ScatterGather(recipients, payload, timeout) Result[[]Data]

// Chain operations
result :=
  Request(agentA, queryA, 5*time.Second).
  AndThen(dataA =>
    Request(agentB, queryB(dataA), 5*time.Second).
    AndThen(dataB =>
      Ok(combine(dataA, dataB))
    )
  )
```

### With L1: Option Type

Receive operations return Option:

```go
Receive() Option[Message]
ReceiveWhere(predicate) Option[Message]
Peek() Option[Message]

// Handle absence
msg := Receive().
  OrElse(defaultMessage).
  AndThen(m => {
    process(m)
    return m
  })
```

### With L4: Effect Isolation

Message operations are effects:

```go
type MessageIO[T] IO[T]

SendIO(recipient, payload) MessageIO[MessageId]
RequestIO(recipient, payload, timeout) MessageIO[Result[Data]]
ReceiveIO() MessageIO[Option[Message]]

// Compose message operations
workflow :=
  SendIO(agentA, taskA).
  FlatMap(_ =>
    RequestIO(agentB, query, 5*time.Second).
    FlatMap(result =>
      PublishIO(Topic("results"), result)
    )
  )

// Execute at boundary
workflow.Run()
```

---

## Quality Metrics

```yaml
reliability:
  score: 0.999
  measurement: "% of messages delivered successfully"
  target: "≥ 0.999 (99.9%)"
  failures: "0.1% timeout, loss, dead letter"

latency_p50:
  score: 8ms
  measurement: "Median send-to-receive time"
  target: "≤ 10ms"

latency_p99:
  score: 85ms
  measurement: "99th percentile latency"
  target: "≤ 100ms"

ordering:
  score: 0.98
  measurement: "% of in-order delivery (per sender-receiver)"
  target: "≥ 0.95"
  note: "Cross-sender ordering not guaranteed"

dedup_accuracy:
  score: 0.995
  measurement: "% of duplicate messages caught"
  target: "≥ 0.99"
  window: "10,000 message IDs, 1 hour"
```

---

## Anti-Patterns

### ❌ Synchronous Overuse
```go
// Bad: Blocking requests in loop (serial)
for task := range tasks {
  result := Request(worker, task, 5*time.Second)  // ❌ Blocks each iteration
  results = append(results, result)
}
// Takes 5s × len(tasks) sequentially

// Good: Parallel with scatter-gather
results := ScatterGather(workers, tasks, 30*time.Second)
// Takes ~5s total (parallel)
```

### ❌ No Timeout
```go
// Bad: Infinite wait
result := Request(agent, data, timeout=∞)  // ❌ May block forever

// Good: Bounded timeout
result := Request(agent, data, timeout=5*time.Second)
match result {
  Ok(data) => use(data)
  Err("timeout") => useDefault()
  Err(e) => handleError(e)
}
```

### ❌ Ignoring Mailbox Size
```go
// Bad: Send without checking capacity
for i := 0; i < 100000; i++ {
  Send(worker, task(i))  // ❌ May overflow worker's mailbox
}

// Good: Check backpressure
for i := 0; i < 100000; i++ {
  for MailboxSize(worker) > 900 {
    time.Sleep(100*time.Millisecond)  // Wait for drain
  }
  Send(worker, task(i))
}
```

### ❌ Non-Idempotent Handlers
```go
// Bad: Process duplicates incorrectly
func OnMessage(msg Message) {
  balance := getBalance()
  balance += msg.Payload.Amount  // ❌ Double-processes duplicates
  setBalance(balance)
}

// Good: Idempotent with deduplication
func OnMessage(msg Message) {
  if alreadyProcessed(msg.Id) {
    return  // Skip duplicate
  }

  balance := getBalance()
  balance += msg.Payload.Amount
  setBalance(balance)
  markProcessed(msg.Id)
}
```

### ❌ Unbounded Retry
```go
// Bad: Retry forever
attempts := 0
for {
  result := Send(agent, msg)
  if result.IsOk() {
    break
  }
  attempts++
  time.Sleep(1*time.Second)  // ❌ Infinite loop if agent down
}

// Good: Bounded retry with exponential backoff
maxAttempts := 3
for attempt := 0; attempt < maxAttempts; attempt++ {
  result := Send(agent, msg)
  if result.IsOk() {
    return Ok(result)
  }
  backoff := time.Duration(1<<attempt) * time.Second
  time.Sleep(backoff)
}
return Err("max_retries_exceeded")
```

---

## Examples

### Example 1: Task Distribution

```go
// Coordinator sends tasks to workers
func DistributeTasks(tasks []Task, workers []AgentId) {
  for i, task := range tasks {
    worker := workers[i % len(workers)]  // Round-robin
    Send(worker, TaskAssignment{Task: task})
  }
}

// Worker receives and processes
func WorkerLoop() {
  for {
    msg := Receive()
    match msg {
      Some(m) => {
        assignment := m.Payload.(TaskAssignment)
        result := processTask(assignment.Task)
        Publish(Topic("task.completed"), TaskResult{
          TaskId: assignment.Task.Id,
          Result: result,
        })
        Ack(m.Id)
      }
      None => {
        time.Sleep(100*time.Millisecond)
      }
    }
  }
}
```

### Example 2: Request/Reply Pattern

```go
// Client requests data
func GetUserData(userId string) Result[User] {
  return Request(
    AgentId("database-agent"),
    GetUserQuery{UserId: userId},
    timeout=5*time.Second
  ).Map(data => data.(User))
}

// Database agent handles requests
func DatabaseAgent() {
  for {
    msg := Receive().Unwrap()

    if msg.Type == "request" {
      query := msg.Payload.(GetUserQuery)
      user := db.GetUser(query.UserId)

      Reply(msg.Id, user)
    }
  }
}
```

### Example 3: Pub/Sub for Monitoring

```go
// Monitoring agent subscribes to all errors
func MonitoringAgent() {
  Subscribe(Topic("error"), func(data Data) {
    error := data.(ErrorEvent)
    log.Error("Error detected:", error)

    if error.Severity == "critical" {
      alertHuman(error)
    }
  })
}

// Any agent can publish errors
func WorkerAgent() {
  result := riskyOperation()
  match result {
    Ok(_) => {}
    Err(e) => {
      Publish(Topic("error"), ErrorEvent{
        Agent: "worker-1",
        Error: e,
        Severity: "critical",
      })
    }
  }
}
```

### Example 4: Pipeline Processing

```go
// Define pipeline stages
pipeline := Pipeline{
  AgentId("validator"),    // Stage 1: Validate
  AgentId("enricher"),     // Stage 2: Enrich
  AgentId("transformer"),  // Stage 3: Transform
  AgentId("storage"),      // Stage 4: Store
}

// Send data through pipeline
SendToPipeline(pipeline, RawData{...})

// Each stage implementation
func ValidatorStage() {
  for {
    msg := Receive().Unwrap()
    validated := validate(msg.Payload)

    if validated.IsValid {
      Send(AgentId("enricher"), validated.Data)
    } else {
      DeadLetter(msg, "validation_failed")
    }
  }
}

func EnricherStage() {
  for {
    msg := Receive().Unwrap()
    enriched := enrich(msg.Payload)
    Send(AgentId("transformer"), enriched)
  }
}

// ... similar for other stages
```

### Example 5: Scatter-Gather Search

```go
// Search across multiple shards in parallel
func ParallelSearch(query string) Result[[]SearchResult] {
  shards := []AgentId{
    AgentId("shard-1"),
    AgentId("shard-2"),
    AgentId("shard-3"),
  }

  results := ScatterGather(
    shards,
    SearchQuery{Term: query},
    timeout=10*time.Second
  )

  return results.Map(responses => {
    // Merge results from all shards
    combined := []SearchResult{}
    for response := range responses {
      combined = append(combined, response.(SearchResult)...)
    }
    return combined
  })
}
```

### Example 6: Reliable Delivery with Retries

```go
// Send with retry logic
func SendReliable(recipient AgentId, payload Data) Result[Unit] {
  maxAttempts := 3

  for attempt := 0; attempt < maxAttempts; attempt++ {
    msgId := Send(recipient, payload)

    // Wait for ack
    ack := ReceiveWhere(m =>
      m.Type == "ack" &&
      m.CorrelationId.Contains(msgId)
    )

    match ack {
      Some(_) => return Ok(Unit)
      None => {
        // No ack, retry with backoff
        backoff := time.Duration(1<<attempt) * time.Second
        time.Sleep(backoff)
      }
    }
  }

  return Err("max_retries_exceeded")
}
```

---

## Message Complexity Formula

```
MESSAGE_COMPLEXITY :=
  1.0 (base send/receive) +
  1.0 × (if request_reply then 1 else 0) +
  2.0 × (if pub_sub then 1 else 0) +
  3.0 × (if scatter_gather then 1 else 0) +
  1.0 × (if exactly_once then 1 else 0)

Example:
  Using send + pub/sub:
  = 1.0 + 0.0 + 2.0 + 0.0 + 0.0
  = 3.0 (medium complexity)

Target: < 5.0
```

---

## Mental Model

```
Message Protocol = Typed Communication Algebra

SEND ────────────> Fire-and-forget (async, 1 slot)
REQUEST ─────────> Wait for reply (sync, 2-3 slots)
PUBLISH ─────────> Broadcast to topic (async, 1-2 slots)
SCATTER-GATHER ──> Parallel fan-out (sync, 4-6 slots)

Reliability:
  Send → [Network] → Receive → Process → Ack
         ↓ loss        ↓ duplicate  ↓ failure
       Retry     Deduplicate    Dead Letter

Flow Control:
  Mailbox fills → Backpressure → Throttle sender
  Mailbox drains → Resume → Normal rate

Key Insight:
  Explicit messages + Typed protocol + Reliable delivery =
    Traceable, debuggable, composable communication
```
