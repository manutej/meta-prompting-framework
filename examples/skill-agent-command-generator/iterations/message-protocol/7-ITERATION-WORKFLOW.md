# 7-Iteration Meta-Prompting: Message Protocol

> Construct → Deconstruct → Reconstruct through 7 iterations

---

## Iteration 1: Foundation - What is Message Passing?

### Question
What is message passing in multi-agent systems, and why do we need a formal protocol?

### Analysis

**Message Passing** is how agents communicate and coordinate. Unlike shared memory, messages are explicit, traceable, and composable.

**Why Message Passing?**
1. **Decoupling** - Agents don't need to know each other's internals
2. **Traceability** - Messages form audit trail
3. **Asynchrony** - Send/receive can be temporally separated
4. **Distribution** - Works across process/network boundaries

**Core Tension:**
```
Flexibility (rich messages) ↔ Simplicity (minimal protocol)
```

Too flexible = hard to reason about, validate
Too simple = can't express complex interactions

**Message Types:**
```
SEND (fire-and-forget)
├─ No reply expected
├─ Best for notifications
└─ Example: "Task completed"

REQUEST/REPLY (synchronous)
├─ Caller waits for response
├─ Best for queries
└─ Example: "What is state?" → "State is X"

PUBLISH/SUBSCRIBE (multicast)
├─ One-to-many broadcast
├─ Best for events
└─ Example: "Error occurred" → [listeners]

STREAM (continuous)
├─ Sequence of messages
├─ Best for data feeds
└─ Example: Metrics stream
```

### Key Insight
Message passing is fundamentally about **making communication explicit and traceable** through structured messages with well-defined semantics.

---

## Iteration 2: Pattern Extraction - Communication Patterns

### Question
What patterns emerge in agent-to-agent communication?

### Pattern Analysis

**6 Fundamental Patterns:**

**1. Point-to-Point**
```
Agent A ──[message]──> Agent B

Properties:
- Direct addressing
- Single recipient
- Guaranteed delivery (or error)
- Order preserved per sender-receiver pair
```

**2. Request/Reply**
```
Agent A ──[request]──> Agent B
Agent A <──[reply]──── Agent B

Properties:
- Correlation ID links request/reply
- Timeout if no reply
- Idempotent retries
- Can chain: A→B→C→B→A
```

**3. Publish/Subscribe**
```
Agent A ──[publish "topic"]──> Topic
                              ↓
                    [Agent B, Agent C, Agent D]

Properties:
- Topic-based routing
- Dynamic subscriptions
- No knowledge of subscribers
- Fire-and-forget
```

**4. Broadcast**
```
Agent A ──[broadcast]──> [All agents in scope]

Properties:
- No targeting (everyone receives)
- Scope can be filtered (e.g., "all workers")
- Order not guaranteed across recipients
```

**5. Pipeline**
```
Agent A ──> Agent B ──> Agent C ──> Agent D

Properties:
- Sequential processing
- Each stage transforms message
- Backpressure propagates upstream
- Example: Data processing pipeline
```

**6. Scatter-Gather**
```
Agent A ──[scatter]──> [B, C, D]
Agent A <──[gather]──── [B, C, D]

Properties:
- Parallel fan-out
- Collect all responses
- Timeout for stragglers
- Partial results possible
```

**Pattern Composition:**

| Pattern | Complexity | Coupling | Use Case |
|---------|------------|----------|----------|
| Point-to-Point | Low | High | Direct communication |
| Request/Reply | Medium | Medium | Queries, commands |
| Pub/Sub | Medium | Low | Event notification |
| Broadcast | Low | Very Low | Announcements |
| Pipeline | High | Medium | Data transformation |
| Scatter-Gather | High | Low | Parallel aggregation |

### Key Insight
**Patterns trade coupling for flexibility.** Point-to-point is simple but tightly coupled; pub/sub is flexible but requires topic management.

---

## Iteration 3: Cognitive Load - Message Complexity

### Question
How does message passing affect agent cognitive load?

### Load Analysis

**Working Memory Slots (7±2 bound):**

```
SEND MESSAGE (1 slot)
├─ Construct message
├─ Address recipient
└─ Send (fire-and-forget)

REQUEST/REPLY (2-3 slots)
├─ Construct request
├─ Track correlation ID
├─ Wait for reply (timeout)
└─ Match reply to request

PUB/SUB (1-2 slots)
├─ Publish: Emit to topic (1 slot)
└─ Subscribe: Register handler (2 slots, callback tracking)

SCATTER-GATHER (4-6 slots) ⚠️ HIGH LOAD
├─ Fan out to N recipients
├─ Track N correlation IDs
├─ Aggregate N responses
├─ Handle partial failures
└─ Implement timeout logic
```

**Message Composition Overhead:**

```
Simple message: 0.5 slots
├─ {type: "notify", data: value}

Structured message: 1.5 slots
├─ {type, correlationId, timestamp, sender, data, metadata}

Complex message: 3+ slots
├─ Nested structure
├─ Multiple correlation IDs
├─ State machine tracking
└─ Error handling
```

**Failure Mode:**
When message tracking exceeds 4 slots, agents enter **cognitive overload**:
- Lose track of pending requests
- Correlation ID mismatches
- Duplicate processing
- Forgotten subscriptions

### Key Insight
**Keep messages simple.** Use fire-and-forget where possible, request/reply sparingly, scatter-gather only when necessary.

---

## Iteration 4: Message Grammar - Formal Protocol

### Question
What is the minimal grammar for message passing?

### Grammar Definition

```
MESSAGE := {
  id: MessageId,           // Unique identifier
  type: MessageType,       // send | request | reply | publish
  sender: AgentId,         // Who sent it
  recipient: Target,       // Where it goes
  correlationId?: MessageId, // Links request/reply
  timestamp: Time,         // When sent
  payload: Data,           // Message content
  metadata?: Metadata      // Optional context
}

TARGET := AgentId | Topic | Broadcast

# Core Operations

Send(recipient: Target, payload: Data) → MessageId
  Post: Message in recipient's mailbox
  Effect: Fire-and-forget

Request(recipient: AgentId, payload: Data, timeout: Duration) → Result[Data]
  Post: Blocks until reply or timeout
  Effect: Synchronous call
  Error: Timeout, recipient not found

Reply(correlationId: MessageId, payload: Data) → Unit
  Pre: correlationId exists in pending requests
  Post: Reply delivered to requester
  Effect: Unblocks waiting request

Publish(topic: Topic, payload: Data) → Unit
  Post: Message sent to all subscribers
  Effect: Multicast notification

Subscribe(topic: Topic, handler: Fn[Data → Unit]) → SubscriptionId
  Post: Handler registered for topic
  Effect: Future publishes trigger handler

Unsubscribe(subscriptionId: SubscriptionId) → Unit
  Post: Handler removed
  Effect: No longer receive topic messages

# Mailbox Operations

Receive() → Option[Message]
  Returns: Next message in mailbox, or None

ReceiveWhere(predicate: Fn[Message → Bool]) → Option[Message]
  Returns: First message matching predicate

Peek() → Option[Message]
  Returns: Next message without removing
```

**Message Ordering Guarantees:**

```
# Per sender-receiver pair, messages are ordered
Send(A, B, m1)
Send(A, B, m2)
=> B receives m1 before m2

# Across different senders, no order guarantee
Send(A, C, m1)
Send(B, C, m2)
=> C may receive m2 before m1
```

**Reliability Semantics:**

```
At-Most-Once: Send may fail silently (fast, unreliable)
At-Least-Once: Send retries until ack (may duplicate)
Exactly-Once: Send with deduplication (expensive, reliable)

Default: At-Least-Once with idempotent handlers
```

### Key Insight
Message protocol is a **typed communication algebra** with send/receive operations and ordering/reliability guarantees.

---

## Iteration 5: Temporal Dynamics - Message Flow Over Time

### Question
How do messages flow through the system over time?

### Temporal Analysis

**Message Lifecycle:**

```
T0: CREATION
├─ Agent constructs message
├─ Assigns unique ID
├─ Sets metadata (timestamp, sender)
└─ Message in "created" state

T1: SEND
├─ Message enqueued for delivery
├─ Routing decision (direct, topic, broadcast)
├─ Message in "in-flight" state
└─ Delivery attempt

T2: RECEIVE
├─ Message arrives at recipient mailbox
├─ Message in "delivered" state
└─ Awaits processing

T3: PROCESS
├─ Recipient dequeues message
├─ Handler executes
├─ May generate reply
└─ Message in "processed" state

T4: COMPLETE
├─ Message acknowledged (if reliable)
├─ Correlation resolved (if request/reply)
├─ Message in "completed" state
└─ Can be garbage collected
```

**Temporal Patterns:**

```
SYNCHRONOUS (Request/Reply)
Timeline:
  T0: A sends request
  T1: B receives request
  T2: B processes request
  T3: B sends reply
  T4: A receives reply
  A blocks from T0 to T4

ASYNCHRONOUS (Send)
Timeline:
  T0: A sends message
  T1: A continues (doesn't wait)
  T2: B receives message (whenever)
  T3: B processes message
  A and B timelines independent

STREAMING
Timeline:
  T0: Producer sends m1
  T1: Consumer receives m1, sends ack
  T2: Producer sends m2
  T3: Consumer receives m2, sends ack
  ...
  Continuous flow with backpressure
```

**Message Accumulation:**

```
Mailbox Size Over Time:
  │
  │     Burst
  │      ╱╲
  │     ╱  ╲
  │    ╱    ╲_____ Steady state
  │___╱
  │─────────────────────────────> Time
  Send Rate > Process Rate = Backlog
  Send Rate < Process Rate = Draining
  Send Rate = Process Rate = Equilibrium
```

**Backpressure Handling:**

```
if mailbox.Size() > threshold:
  # Option 1: Throttle sender
  SendBackpressure(sender, "slow down")

  # Option 2: Drop messages
  DropOldest()  # or DropNewest()

  # Option 3: Block sender
  BlockUntil(mailbox.Size() < threshold)
```

### Key Insight
**Messages have lifecycle states.** Monitor mailbox sizes to detect backpressure and apply flow control.

---

## Iteration 6: Failure Modes - Error Handling

### Question
What can go wrong with message passing, and how do we handle it?

### Failure Analysis

**8 Common Failures:**

**1. Message Loss**
```
Cause: Network partition, agent crash
Detection: Timeout on request/reply
Recovery:
  - Retry with exponential backoff
  - Use reliable delivery (at-least-once)
  - Idempotent handlers
```

**2. Duplicate Messages**
```
Cause: Retry after network delay (not actual loss)
Detection: Duplicate message ID
Recovery:
  - Deduplication cache (track seen IDs)
  - Idempotent operations
  - Version numbers on state
```

**3. Out-of-Order Delivery**
```
Cause: Multiple paths, variable latency
Detection: Sequence numbers
Recovery:
  - Reorder buffer
  - Reject out-of-order (if strict ordering needed)
  - Accept unordered (if commutative ops)
```

**4. Mailbox Overflow**
```
Cause: Send rate > process rate
Detection: Mailbox size threshold
Recovery:
  - Backpressure to sender
  - Drop messages (with logging)
  - Spawn more workers
```

**5. Dead Letter (No Recipient)**
```
Cause: Recipient doesn't exist, topic has no subscribers
Detection: Delivery failure
Recovery:
  - Dead letter queue
  - Alert sender
  - Retry with exponential backoff
```

**6. Timeout (No Reply)**
```
Cause: Recipient too slow, crashed, or overwhelmed
Detection: Request timeout expires
Recovery:
  - Return timeout error to caller
  - Retry (if idempotent)
  - Escalate to human
```

**7. Correlation Mismatch**
```
Cause: Reply without matching request
Detection: Unknown correlation ID
Recovery:
  - Log warning
  - Drop orphaned reply
  - Garbage collect old requests
```

**8. Poison Message**
```
Cause: Malformed message, handler bug
Detection: Handler throws exception
Recovery:
  - Move to dead letter queue
  - Alert monitoring
  - Skip message (don't retry infinitely)
```

**Error Handling Matrix:**

| Failure | Detection | Recovery | Cost |
|---------|-----------|----------|------|
| Loss | Timeout | Retry | Medium |
| Duplicate | ID cache | Dedupe | Low |
| Out-of-order | Sequence # | Reorder | Medium |
| Overflow | Mailbox size | Backpressure | High |
| No recipient | Delivery fail | Dead letter | Low |
| Timeout | Timer | Error | Low |
| Mismatch | Unknown ID | Drop | Low |
| Poison | Exception | Quarantine | Medium |

### Key Insight
**Design for failure.** Use timeouts, retries, deduplication, and dead letter queues to handle inevitable message failures.

---

## Iteration 7: Final Synthesis - Optimal Message Protocol

### Synthesis

**OPTIMAL_MESSAGE_PROTOCOL** is a **typed, reliable, traceable communication system**:

```haskell
data MessageProtocol m = MessageProtocol {
  -- Send operations
  send      :: Target → Data → m MessageId,
  request   :: AgentId → Data → Duration → m (Result Data),
  reply     :: MessageId → Data → m Unit,
  publish   :: Topic → Data → m Unit,

  -- Receive operations
  receive      :: m (Option Message),
  receiveWhere :: (Message → Bool) → m (Option Message),
  peek         :: m (Option Message),

  -- Subscription
  subscribe   :: Topic → (Data → m Unit) → m SubscriptionId,
  unsubscribe :: SubscriptionId → m Unit,

  -- Reliability
  ack      :: MessageId → m Unit,
  nack     :: MessageId → Reason → m Unit,
  deadLetter :: Message → Reason → m Unit,

  -- Observability
  mailboxSize :: m Int,
  pending     :: m [MessageId],
  topics      :: m [Topic]
}
```

**Design Principles:**

1. **Explicit Over Implicit**:
   ```
   # Explicit sender/recipient in message
   message = {sender: "A", recipient: "B", ...}

   # Not implicit (e.g., thread-local context)
   ```

2. **Idempotent Handlers**:
   ```
   # Handler can be called multiple times safely
   def onMessage(msg):
     if isDuplicate(msg.id):
       return  # Already processed
     process(msg)
     recordProcessed(msg.id)
   ```

3. **Correlation Tracking**:
   ```
   # Request
   reqId = Request(target, data, timeout=5s)

   # Reply references request
   Reply(correlationId=reqId, data=result)

   # System matches them automatically
   ```

4. **Bounded Mailboxes**:
   ```
   mailbox_capacity = 1000

   if mailbox.size() > 900:
     emit_backpressure_signal()

   if mailbox.size() >= 1000:
     drop_oldest() OR block_sender()
   ```

5. **Dead Letter Queue**:
   ```
   # Failed message goes to DLQ
   if delivery_attempts > 3:
     DeadLetter(message, reason="max_retries")
     alert_monitoring()
   ```

6. **Observable Messaging**:
   ```
   # Every message operation emits event
   Send(msg) → emit("message.sent", msg.id, sender, recipient)
   Receive(msg) → emit("message.received", msg.id, recipient)
   Process(msg) → emit("message.processed", msg.id, duration)
   ```

**Quality Metrics:**

```yaml
message_protocol_quality:
  reliability: ≥0.999  # 99.9% delivery success
  latency_p50: ≤10ms   # Median send-to-receive
  latency_p99: ≤100ms  # Tail latency
  ordering: ≥0.95      # % of ordered delivery (per sender-receiver)
  dedup_accuracy: ≥0.99 # % of duplicates caught
```

**Complexity Formula:**

```
MESSAGE_COMPLEXITY :=
  1.0 (base send/receive) +
  1.0 × (if request_reply then 1 else 0) +
  2.0 × (if pub_sub then 1 else 0) +
  3.0 × (if scatter_gather then 1 else 0) +
  1.0 × (if exactly_once then 1 else 0)

Target: < 5.0

Recommendation:
  - Use send (fire-and-forget) where possible
  - Request/reply for queries
  - Pub/sub for events
  - Avoid scatter-gather unless necessary
```

**Self-Reference:**

This message protocol **describes itself using messages**:
- Meta-messages: Protocol evolution happens via messages
- Bootstrap: First message establishes protocol
- Homoiconic: Messages about messages use same protocol

The system is **self-describing** - protocol specification is itself a message.

---

## Final Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MESSAGE PROTOCOL                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  SEND OPERATIONS                                   │    │
│  │  • Send: Fire-and-forget                          │    │
│  │  • Request: Wait for reply                        │    │
│  │  • Publish: Topic broadcast                       │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  ROUTING & DELIVERY                                │    │
│  │  • Point-to-Point: Direct to agent                 │    │
│  │  • Topic-Based: Route by subscription             │    │
│  │  • Broadcast: All agents in scope                 │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  MAILBOX (Bounded Queue)                          │    │
│  │  • Capacity: 1000 messages                        │    │
│  │  • Backpressure at 90%                            │    │
│  │  • Overflow: Drop oldest or block                 │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RECEIVE OPERATIONS                                │    │
│  │  • Receive: Dequeue next message                  │    │
│  │  • ReceiveWhere: Selective receive                │    │
│  │  • Peek: Look without removing                    │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RELIABILITY                                       │    │
│  │  • Ack/Nack: Confirm processing                   │    │
│  │  • Retry: Exponential backoff (3 attempts)        │    │
│  │  • Dedupe: Cache seen message IDs                 │    │
│  │  • Dead Letter: Failed messages quarantined       │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  GUARANTEES:                                                │
│  • At-Least-Once delivery (with idempotent handlers)        │
│  • FIFO ordering per sender-receiver pair                   │
│  • Correlation tracking for request/reply                   │
│  • Observable (all operations emit events)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Meta-Learning

What did we learn about **message protocols** through meta-prompting?

1. **Explicit communication** - Messages make interactions traceable
2. **Six core patterns** - Point-to-point, request/reply, pub/sub, broadcast, pipeline, scatter-gather
3. **Cognitive load varies** - Fire-and-forget is cheap, scatter-gather is expensive
4. **Typed protocol** - Messages have schema, operations, guarantees
5. **Temporal dynamics** - Messages have lifecycle, mailboxes fill/drain
6. **Failure is normal** - Design for timeouts, retries, deduplication, dead letters
7. **Observable by default** - Every message operation emits event

The pattern: **Construct typed messages → Route to recipients → Queue in mailbox → Process with handlers → Handle failures gracefully → Emit observability events**
