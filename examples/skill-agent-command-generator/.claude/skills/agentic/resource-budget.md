# Resource Budget Skill

> Prevent unbounded consumption through predictive enforcement

**Level**: L4 (Effect management with resource constraints)
**Dependencies**: L2-result-type, L1-option-type
**Cognitive Load**: 1.5-2.5 slots (mostly automatic)

---

## Context

Multi-agent systems can consume unbounded resources (tokens, time, cost) leading to runaway costs and system failures. Resource budgeting provides **lifecycle management** from allocation through enforcement to prevent unbounded consumption while maximizing work completion.

**Problem**:
- Agents can enter infinite loops
- Token/cost consumption is unbounded
- No visibility into resource usage
- Hard to debug expensive operations

**Solution**:
- Allocate budgets upfront
- Track consumption automatically
- Enforce with progressive warnings
- Report usage for attribution
- Release unused resources

---

## Capability

### Phase 1: Allocation

Define and assign budgets to agents.

```go
type BudgetSpec struct {
  Tokens      Option[int]  // LLM tokens limit
  TimeSeconds Option[int]  // Wall-clock time limit
  CostCents   Option[int]  // Monetary cost limit
}

// Allocate budget to agent
Allocate(spec BudgetSpec, agentId string) Result[Allocation]

// Example usage
budget := BudgetSpec{
  Tokens: Some(50000),      // 50k tokens
  TimeSeconds: Some(300),   // 5 minutes
  CostCents: Some(500),     // $5.00
}

allocation := Allocate(budget, "worker-1")
```

**Allocation Strategies:**

```go
// Equal: Each agent gets same budget
func AllocateEqual(total BudgetSpec, agents []string) []Allocation {
  perAgent := BudgetSpec{
    Tokens: total.Tokens.Map(t => t / len(agents)),
    TimeSeconds: total.TimeSeconds.Map(t => t / len(agents)),
    CostCents: total.CostCents.Map(c => c / len(agents)),
  }

  return agents.Map(agent => Allocate(perAgent, agent).Unwrap())
}

// Proportional: Based on estimated work
func AllocateProportional(
  total BudgetSpec,
  agents map[string]float64,  // agent → weight
) []Allocation {
  totalWeight := sum(agents.values())

  return agents.Map((agent, weight) =>
    Allocate(BudgetSpec{
      Tokens: total.Tokens.Map(t => int(t * weight / totalWeight)),
      // ... similar for time and cost
    }, agent).Unwrap()
  )
}

// Dynamic Pool: Shared budget, allocate on demand
func AllocateFromPool(agentId string, request BudgetSpec) Result[Allocation] {
  pool := ReadLocal("budget_pool").Unwrap()

  if pool.Available < request {
    return Err("Insufficient budget in pool")
  }

  pool.Allocated[agentId] = request
  pool.Available = pool.Available - request
  WriteLocal("budget_pool", pool)

  return Ok(Allocation{AgentId: agentId, Budget: request})
}
```

---

### Phase 2: Tracking

Automatically measure consumption in real-time.

```go
type Usage struct {
  TokensUsed  int
  TimeUsed    int  // seconds
  CostAccrued int  // cents
}

// Track usage (called automatically after each operation)
Track(agentId string, usage Usage) Unit

// Check current status
CheckStatus(agentId string) BudgetStatus

type BudgetStatus struct {
  Allocated  BudgetSpec
  Consumed   Usage
  Remaining  BudgetSpec
  Percentage float64  // 0.0 to 1.0
  Status     StatusLevel  // Green/Yellow/Orange/Red/Black
}

// Example usage
status := CheckStatus("worker-1")
println(f"Budget: {status.Percentage * 100:.1f}% used")
println(f"Status: {status.Status}")
println(f"Remaining: {status.Remaining.Tokens} tokens")
```

**Automatic Tracking:**

```go
// Operations automatically report usage
func CallLLM(prompt string) Result[string] {
  startTime := time.Now()

  // Make LLM call
  response, tokens := llm.Complete(prompt)

  // Automatically track
  usage := Usage{
    TokensUsed: tokens,
    TimeUsed: int(time.Since(startTime).Seconds()),
    CostAccrued: CalculateCost(tokens),
  }
  Track(GetCurrentAgent(), usage)

  return Ok(response)
}
```

**Status Levels:**

```go
func DetermineStatus(percentage float64) StatusLevel {
  switch {
    case percentage < 0.50: return Green
    case percentage < 0.75: return Yellow
    case percentage < 0.90: return Orange
    case percentage < 0.95: return Red
    default: return Black
  }
}
```

---

### Phase 3: Enforcement

Control execution based on budget consumption.

```go
type Action = Continue | Warn | Throttle | Degrade | Stop

// Determine what action to take
Enforce(agentId string) Action

// Apply throttling
Throttle(agentId string, rate float64) Unit

// Example enforcement
func EnforceBeforeOperation(agentId string) Result[Unit] {
  action := Enforce(agentId)

  match action {
    Continue => return Ok(Unit)

    Warn => {
      log.Warn(f"Agent {agentId} at 50% budget")
      return Ok(Unit)
    }

    Throttle => {
      log.Warn(f"Agent {agentId} at 75% budget, throttling")
      time.Sleep(1 * time.Second)  // Rate limit
      return Ok(Unit)
    }

    Degrade => {
      log.Warn(f"Agent {agentId} at 90% budget, degrading quality")
      UseFasterModel()  // Trade quality for cost
      return Ok(Unit)
    }

    Stop => {
      log.Error(f"Agent {agentId} exceeded budget, stopping")
      return Err("Budget exhausted")
    }
  }
}
```

**Progressive Enforcement:**

```go
// Enforcement gets stricter as budget depletes
func Enforce(agentId string) Action {
  status := CheckStatus(agentId)

  switch status.Status {
    case Green:
      return Continue

    case Yellow:
      emitEvent("budget.warning", agentId, status)
      return Warn

    case Orange:
      emitEvent("budget.throttle", agentId, status)
      Throttle(agentId, 0.5)  // 50% rate reduction
      return Throttle

    case Red:
      emitEvent("budget.degrade", agentId, status)
      // Disable non-essential features
      DisableFeatures(agentId, ["detailed_analysis", "examples"])
      return Degrade

    case Black:
      emitEvent("budget.exhausted", agentId, status)
      return Stop
  }
}
```

---

### Phase 4: Reporting

Analyze and forecast usage patterns.

```go
type UsageReport struct {
  AgentId       string
  Allocated     BudgetSpec
  Consumed      Usage
  Efficiency    float64  // Work completed per unit consumed
  Forecast      Prediction
  Anomalies     []Anomaly
}

// Generate report
Report(agentId string) UsageReport

// Forecast remaining work
Forecast(agentId string) Prediction

type Prediction struct {
  EstimatedTotal    Usage
  ProjectedOverage  Option[Usage]
  TimeToExhaustion  Option[int]  // seconds
  ConfidenceLevel   float64      // 0.0 to 1.0
}

// Example usage
report := Report("worker-1")
println(f"Efficiency: {report.Efficiency:.2f} tasks/1k tokens")

if overage := report.Forecast.ProjectedOverage; overage.IsSome() {
  println(f"WARNING: Projected to exceed by {overage.Unwrap().Tokens} tokens")
}
```

**Consumption Pattern Detection:**

```go
// Detect consumption patterns for prediction
func AnalyzePattern(agentId string) ConsumptionPattern {
  history := GetUsageHistory(agentId)

  // Fit consumption curve
  if IsLinear(history) {
    rate := history.Last().TokensUsed / history.Last().TimeUsed
    return LinearPattern{Rate: rate}
  }

  if IsExponential(history) {
    growthRate := CalculateGrowthRate(history)
    return ExponentialPattern{GrowthRate: growthRate}
  }

  if IsBursty(history) {
    return BurstyPattern{Intervals: DetectBursts(history)}
  }

  return UnknownPattern{}
}

// Forecast based on pattern
func Forecast(agentId string) Prediction {
  pattern := AnalyzePattern(agentId)
  status := CheckStatus(agentId)

  match pattern {
    LinearPattern(rate) => {
      remaining := status.Remaining.Tokens.Unwrap()
      timeLeft := remaining / rate
      return Prediction{
        TimeToExhaustion: Some(timeLeft),
        ConfidenceLevel: 0.85,
      }
    }

    ExponentialPattern(growth) => {
      // Calculate when exponential curve hits limit
      timeLeft := log(remaining) / growth
      return Prediction{
        TimeToExhaustion: Some(timeLeft),
        ConfidenceLevel: 0.65,  // Lower confidence for exponential
      }
    }

    // ... other patterns
  }
}
```

---

### Phase 5: Release

Return unused budget and finalize accounting.

```go
type ReleasedBudget struct {
  AgentId      string
  Allocated    BudgetSpec
  Consumed     Usage
  Unused       BudgetSpec
  FinalReport  UsageReport
}

// Release budget when agent completes
Release(agentId string) Result[ReleasedBudget]

// Example usage
func AgentComplete(agentId string) {
  released := Release(agentId).Unwrap()

  // Return unused to pool
  if pool := ReadLocal("budget_pool"); pool.IsSome() {
    pool.Available = pool.Available + released.Unused
    WriteLocal("budget_pool", pool)
  }

  // Log final usage
  log.Info(f"Agent {agentId} used {released.Consumed.Tokens} of {released.Allocated.Tokens} tokens")
}
```

---

## Constraints

### 1. Budget Boundaries
```yaml
rule: "All budgets must be non-negative and finite"
examples:
  ✅ valid: "BudgetSpec{Tokens: Some(1000)}"
  ❌ invalid: "BudgetSpec{Tokens: Some(-100)}"
  ❌ invalid: "BudgetSpec{Tokens: Some(∞)}"
```

### 2. Allocation ≤ Total
```yaml
rule: "Sum of allocations cannot exceed total budget"
check: |
  sum(allocations) ≤ total_budget
enforcement: "Reject allocation if would exceed total"
```

### 3. Tracking Overhead
```yaml
rule: "Budget tracking uses < 1% of total budget"
measurement: |
  overhead = tracking_operations / total_operations
  assert overhead < 0.01
```

### 4. Enforcement Latency
```yaml
rule: "Enforcement acts within 1 second of threshold"
measurement: |
  latency = time(threshold_crossed) - time(enforcement_triggered)
  assert latency < 1.0s
```

### 5. Minimum Reserve
```yaml
rule: "Each agent guaranteed minimum 5% of equal share"
formula: |
  min_per_agent = total_budget * 0.05 / num_agents
  allocation ≥ min_per_agent
```

---

## Composition

### With L2: Result Type

Operations that can fail return Result:

```go
Allocate(spec, agent) Result[Allocation]  // May fail if pool exhausted
Release(agent) Result[ReleasedBudget]     // May fail if agent not found

// Chain operations
result :=
  Allocate(budget, "agent-1").
  AndThen(allocation =>
    RunAgent("agent-1").
    AndThen(output =>
      Release("agent-1").
      Map(released => (output, released))
    )
  )

match result {
  Ok((output, released)) =>
    println(f"Success: {released.Consumed.Tokens} tokens")
  Err(e) =>
    println(f"Failed: {e}")
}
```

### With L1: Option Type

Optional budget components:

```go
spec := BudgetSpec{
  Tokens: Some(10000),
  TimeSeconds: None,  // No time limit
  CostCents: Some(100),
}

// Handle optional limits
func CheckLimit[T](limit Option[T], used T) bool {
  return limit.Map(l => used < l).OrElse(true)
}
```

### With L4: Effect Isolation

Budget operations are effects:

```go
type BudgetIO[T] IO[T]

AllocateIO(spec, agent) BudgetIO[Result[Allocation]]
TrackIO(agent, usage) BudgetIO[Unit]
ReleaseIO(agent) BudgetIO[Result[ReleasedBudget]]

// Compose budget lifecycle
workflow :=
  AllocateIO(budget, agent).
  FlatMap(allocation =>
    RunWorkIO(agent).
    FlatMap(result =>
      ReleaseIO(agent).
      Map(_ => result)
    )
  )

// Execute at boundary
result := workflow.Run()
```

---

## Quality Metrics

```yaml
utilization:
  score: 0.87
  measurement: "consumed / allocated"
  target: "≥ 0.80"
  reasoning: "High utilization = efficient allocation"

accuracy:
  score: 0.94
  measurement: "% of tracking within 10% of actual"
  target: "≥ 0.90"
  validation: "Compare tracked vs ground truth"

responsiveness:
  score: 0.97
  measurement: "% of enforcement actions within 1s"
  target: "≥ 0.95"
  latency_p50: "0.1s"
  latency_p99: "0.8s"

fairness:
  score: 0.89
  measurement: "1 - variance(allocations) / mean(allocations)"
  target: "≥ 0.85"
  current_variance: "11%"
```

---

## Anti-Patterns

### ❌ No Budget Limits
```go
// Bad: Unbounded execution
func RunTask() {
  for {
    // Infinite loop, no budget check
    processItem()
  }
}

// Good: Check budget before each iteration
func RunTask(agentId string) Result[Unit] {
  for {
    Enforce(agentId)??  // Will stop if budget exhausted
    processItem()
  }
  return Ok(Unit)
}
```

### ❌ Manual Tracking
```go
// Bad: Manual tracking (error-prone, forgotten)
func CallLLM(prompt string) {
  response := llm.Complete(prompt)
  Track(GetAgent(), Usage{TokensUsed: 1000})  // ❌ Manual, hardcoded
  return response
}

// Good: Automatic tracking in wrapper
func CallLLM(prompt string) Result[string] {
  return MeasuredOperation(func() string {
    return llm.Complete(prompt)
  })
}
```

### ❌ Hard Stop Without Warning
```go
// Bad: Immediate stop at 100%
if percentage >= 1.0 {
  panic("Budget exceeded")  // ❌ No warning, no degradation
}

// Good: Progressive enforcement
action := Enforce(agentId)
match action {
  Warn => log.Warn("50% budget used")
  Throttle => time.Sleep(delay)
  Degrade => useFasterModel()
  Stop => return Err("Budget exhausted")
  _ => continue
}
```

### ❌ Ignoring Forecasts
```go
// Bad: Wait until exhaustion
if CheckStatus(agent).Percentage >= 1.0 {
  stop()
}

// Good: Act on prediction
forecast := Forecast(agent)
if overage := forecast.ProjectedOverage; overage.IsSome() {
  log.Warn("Projected to exceed budget, throttling now")
  Throttle(agent, 0.5)
}
```

---

## Examples

### Example 1: Multi-Agent Workflow with Budgets

```go
func RunWorkflow(tasks []Task) Result[[]Output] {
  // Allocate budgets
  totalBudget := BudgetSpec{
    Tokens: Some(100000),
    TimeSeconds: Some(600),
    CostCents: Some(1000),
  }

  // Proportional allocation based on task complexity
  weights := tasks.Map(t => t.EstimatedComplexity)
  allocations := AllocateProportional(totalBudget, weights)

  // Run agents with budgets
  outputs := []Output{}
  for (task, allocation) := range zip(tasks, allocations) {
    agent := SpawnAgent(task, allocation)

    // Monitor budget
    for {
      status := CheckStatus(agent.Id)

      if status.Status == Black {
        log.Error(f"Agent {agent.Id} exceeded budget")
        break
      }

      if agent.IsComplete() {
        output := agent.GetOutput()
        outputs = append(outputs, output)
        break
      }

      time.Sleep(1 * time.Second)
    }

    // Release budget
    released := Release(agent.Id).Unwrap()
    log.Info(f"Agent used {released.Consumed.Tokens} tokens")
  }

  return Ok(outputs)
}
```

### Example 2: Dynamic Pool with Borrowing

```go
// Shared budget pool
type BudgetPool struct {
  Total     BudgetSpec
  Allocated map[string]BudgetSpec
  Available BudgetSpec
}

func RequestBudget(agentId string, requested BudgetSpec) Result[Allocation] {
  pool := ReadLocal("budget_pool").Unwrap()

  // Check if available
  if pool.Available.Tokens.Unwrap() >= requested.Tokens.Unwrap() {
    // Allocate directly
    pool.Available.Tokens = Some(
      pool.Available.Tokens.Unwrap() - requested.Tokens.Unwrap()
    )
    pool.Allocated[agentId] = requested
    WriteLocal("budget_pool", pool)

    return Ok(Allocation{AgentId: agentId, Budget: requested})
  }

  // Try borrowing from idle agents
  borrowed := BorrowFromIdle(requested.Tokens.Unwrap())
  if borrowed.IsSome() {
    pool.Allocated[agentId] = requested
    WriteLocal("budget_pool", pool)
    return Ok(Allocation{AgentId: agentId, Budget: requested})
  }

  return Err("Insufficient budget available")
}

func BorrowFromIdle(needed int) Option[int] {
  pool := ReadLocal("budget_pool").Unwrap()
  borrowed := 0

  for (agentId, allocation) := range pool.Allocated {
    status := CheckStatus(agentId)

    // If agent idle and has excess, borrow
    if status.Status == Green && status.Percentage < 0.25 {
      excess := allocation.Tokens.Unwrap() * 0.5  // Borrow 50% of their budget
      borrowed += excess

      // Reduce their allocation
      allocation.Tokens = Some(allocation.Tokens.Unwrap() - excess)
      pool.Allocated[agentId] = allocation

      if borrowed >= needed {
        WriteLocal("budget_pool", pool)
        return Some(borrowed)
      }
    }
  }

  return None
}
```

### Example 3: Predictive Throttling

```go
func PredictiveEnforcement(agentId string) {
  forecast := Forecast(agentId)

  // If projected to exceed, throttle preemptively
  if overage := forecast.ProjectedOverage; overage.IsSome() {
    log.Warn(f"Agent {agentId} projected to exceed by {overage.Unwrap().Tokens} tokens")

    // Calculate throttle rate to stay within budget
    status := CheckStatus(agentId)
    targetRate := status.Remaining.Tokens.Unwrap() /
                  forecast.TimeToExhaustion.Unwrap()
    currentRate := status.Consumed.TokensUsed / status.Consumed.TimeUsed

    throttle := targetRate / currentRate
    log.Info(f"Applying throttle: {throttle:.2f}x")
    Throttle(agentId, throttle)
  }
}

// Run predictive enforcement periodically
func MonitorBudgets() {
  for {
    agents := GetActiveAgents()
    for agent := range agents {
      PredictiveEnforcement(agent.Id)
    }
    time.Sleep(5 * time.Second)
  }
}
```

---

## Budget Complexity Formula

```
BUDGET_OVERHEAD :=
  0.1 × num_agents +
  1.0 × (if tracking_per_operation then 1 else 0) +
  2.0 × (if dynamic_reallocation then 1 else 0) +
  5.0 × (if auction_based then 1 else 0)

Target: BUDGET_OVERHEAD < 5

Recommendation:
- Agent-level tracking (not per-operation)
- Static allocation (not dynamic)
- Simple strategies (not auction)
```

---

## Mental Model

```
Resource Budget = Lifecycle State Machine

ALLOCATE ──────> Budget Reserved
    │
    ▼
  TRACK ────────> Consumption Measured (automatic)
    │
    ▼
 ENFORCE ───────> Action Taken (progressive)
    │             ├─ Green: Continue
    │             ├─ Yellow: Warn
    │             ├─ Orange: Throttle
    │             ├─ Red: Degrade
    │             └─ Black: Stop
    ▼
 REPORT ────────> Usage Analyzed
    │             ├─ Efficiency metrics
    │             └─ Forecasts
    ▼
 RELEASE ───────> Budget Returned

Key Insight:
  Automatic tracking + Predictive enforcement =
    Safety without cognitive overhead
```
