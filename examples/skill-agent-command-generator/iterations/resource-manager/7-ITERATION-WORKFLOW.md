# 7-Iteration Meta-Prompting: Resource Manager Agent

> Construct → Deconstruct → Reconstruct through 7 iterations

---

## Iteration 1: Foundation - Why Centralize Budgeting?

### Question
Why do we need a dedicated agent for resource management?

### Analysis

**Resource Manager** is a specialized agent that centralizes the 5-phase budget lifecycle (allocate/track/enforce/report/release) defined in the resource-budget skill.

**Problem with Distributed Budgeting:**
- Each agent tracking its own budget = inconsistent global view
- No coordination = agents can collectively exceed total budget
- Hard to rebalance budgets dynamically
- Difficult to forecast total resource consumption

**Benefits of Centralization:**
- **Global view** - See total resource consumption across all agents
- **Dynamic allocation** - Rebalance budgets based on actual usage
- **Enforcement** - Prevent total budget exhaustion
- **Forecasting** - Predict when system will hit limits

**Analogy:**
```
Distributed Budget = Each department manages own budget (can collectively overspend)
Centralized Budget = Finance department manages all budgets (global control)
```

### Key Insight
**Resource Manager provides budget-as-a-service**, enabling centralized control with distributed consumption.

---

## Iteration 2: Pattern Extraction - Manager Responsibilities

### Question
What should Resource Manager do vs delegate?

### Responsibility Analysis

**DOES:**
1. **Budget Allocation**
   - Allocate budgets to agents (equal, proportional, priority-based)
   - Manage shared budget pool
   - Handle dynamic reallocation

2. **Consumption Tracking**
   - Receive usage reports from agents
   - Aggregate total consumption
   - Track per-agent and per-workflow metrics

3. **Enforcement**
   - Monitor budget status (green/yellow/orange/red/black)
   - Emit warnings and alerts
   - Throttle or stop agents at limits

4. **Reporting & Forecasting**
   - Generate usage reports
   - Predict exhaustion times
   - Recommend optimizations

**DOES NOT:**
1. **Direct Measurement** - Agents self-report usage (honor system + auditing)
2. **Business Logic** - Just manages budgets, not tasks
3. **State Storage** - Uses State Keeper for coordination state
4. **Monitoring** - Monitor agent handles observability

**Responsibility Split:**

| Component | Responsibility |
|-----------|----------------|
| **Agent** | Self-report usage after operations |
| **Resource Manager** | Allocate, track, enforce budgets |
| **State Keeper** | Store budget allocations/consumption |
| **Monitor** | Observe budget metrics, alert on thresholds |

### Key Insight
**Resource Manager coordinates, agents self-report.** Trust but verify through auditing.

---

## Iteration 3: Cognitive Load - Offloading Budget Tracking

### Question
How does Resource Manager reduce agent cognitive load?

### Load Analysis

**Before Resource Manager (Each Agent Manages Budget):**
```
Agent Working Memory:
├─ Task logic: 3-4 slots
├─ Budget tracking: 2-3 slots (remaining tokens/time/cost, percentage)
├─ Enforcement logic: 1-2 slots (check status, throttle self)
├─ Reporting: 1 slot (emit metrics)
└─ Total: 7-10 slots ❌ OVERLOAD
```

**After Resource Manager (Centralized):**
```
Agent Working Memory:
├─ Task logic: 3-4 slots
├─ Report usage: 0.5 slots (just send usage after op)
└─ Total: 3.5-4.5 slots ✅ MANAGEABLE

Resource Manager Working Memory:
├─ Track all agents: 6-7 slots (its specialty)
└─ No task logic (focused role)
```

**Delegation Pattern:**
```go
// Before: Agent tracks its own budget
func AgentWithBudget() {
  allocated := GetMyBudget()  // 1 slot
  consumed := GetMyConsumption()  // 1 slot
  remaining := allocated - consumed  // 1 slot
  percentage := consumed / allocated  // 1 slot

  if percentage > 0.75 {  // 1 slot
    throttle()
  }
  // Total: 5 slots just for budgeting
}

// After: Agent reports to Resource Manager
func AgentWithManager() {
  // Do work
  result := doWork()

  // Report usage (automatic)
  ReportUsage(tokens=450, time=2.3s)  // 0.5 slots

  // Resource Manager handles enforcement
}
```

### Key Insight
**Centralization enables automatic tracking.** Agents don't think about budgets, just report usage.

---

## Iteration 4: Resource Manager Grammar - API

### Question
What operations should Resource Manager expose?

### API Definition

```
RESOURCE_MANAGER_API := {
  # Allocation
  AllocateBudget(agent, spec) → Result[Allocation]
  AllocateEqual(agents, total) → Result[[]Allocation]
  AllocateProportional(weights, total) → Result[[]Allocation]
  AllocateFromPool(agent, request) → Result[Allocation]

  # Tracking
  ReportUsage(agent, usage) → Unit
  GetStatus(agent) → BudgetStatus
  GetGlobalStatus() → GlobalBudgetStatus

  # Enforcement
  CheckEnforcement(agent) → Action  # Continue/Warn/Throttle/Degrade/Stop
  Throttle(agent, rate) → Unit
  Stop(agent, reason) → Unit

  # Reallocation
  Rebalance() → Result[[]Allocation]
  BorrowFromPool(agent, amount) → Result[Unit]
  ReturnToPool(agent, amount) → Unit

  # Reporting
  GenerateReport(workflow) → UsageReport
  Forecast(agent) → Prediction
  GetEfficiency(agent) → EfficiencyMetrics

  # Release
  ReleaseBudget(agent) → Result[ReleasedBudget]
}
```

**Budget Status:**
```go
type BudgetStatus struct {
  Agent       AgentId
  Allocated   BudgetSpec  // {tokens, time, cost}
  Consumed    Usage       // {tokens_used, time_used, cost_accrued}
  Remaining   BudgetSpec
  Percentage  float64     // 0.0 to 1.0
  Status      StatusLevel // Green/Yellow/Orange/Red/Black
}

type GlobalBudgetStatus struct {
  TotalAllocated BudgetSpec
  TotalConsumed  Usage
  PoolAvailable  BudgetSpec
  AgentCount     int
  AgentStatuses  []BudgetStatus
}
```

### Key Insight
**API mirrors 5-phase lifecycle** from resource-budget skill: allocate → track → enforce → report → release.

---

## Iteration 5: Temporal Dynamics - Dynamic Reallocation

### Question
How does Resource Manager handle changing resource needs over time?

### Reallocation Analysis

**Static Allocation Problems:**
```
Time:  T0──────────T1──────────T2──────────T3
Agent A: [============Done]
         Allocated 50k tokens, used 20k (40% waste)

Agent B: [========EXHAUSTED!]
         Allocated 50k tokens, used 50k (needs more)
```

**Dynamic Reallocation Solution:**
```
Time:  T0──────────T1──────────T2──────────T3
Agent A: [======]       (releases early)
           ↓
         [Pool]
           ↓
Agent B: [============Done]  (borrows from pool)
```

**Reallocation Strategies:**

**1. Idle Reclamation**
```go
// Detect idle agents, reclaim their budget
for agent := range agents {
  status := GetStatus(agent)
  if status.IsIdle() && status.Percentage < 0.25 {
    // Agent has used <25%, is idle
    reclaimable := status.Remaining * 0.5  // Take 50% of remaining
    ReturnToPool(agent, reclaimable)
  }
}
```

**2. Demand-Based Allocation**
```go
// Allocate more to agents approaching limits
for agent := range agents {
  status := GetStatus(agent)
  if status.Percentage > 0.80 && status.Forecast.WillExceed() {
    // Agent at 80%, projected to exceed
    additional := CalculateNeed(status.Forecast)
    BorrowFromPool(agent, additional)
  }
}
```

**3. Priority-Based**
```go
// Critical agents get priority access to pool
if poolAvailable > 0 {
  criticalAgents := agents.Filter(a => a.Priority == "critical")
  for agent := range criticalAgents {
    if agent.NeedsBudget() {
      AllocateFromPool(agent, agent.Request)
    }
  }
}
```

### Key Insight
**Dynamic reallocation improves utilization** from static ~60% to dynamic ~85%+.

---

## Iteration 6: Failure Modes - Budget Exhaustion

### Question
What happens when total budget is exhausted?

### Exhaustion Handling

**Exhaustion Scenarios:**

**1. Individual Agent Exhaustion**
```yaml
trigger: Agent reaches 100% of allocated budget
action:
  - Mark agent status as Black
  - Stop agent (return error on operations)
  - Alert monitoring
  - Release any unused budget to pool
impact: "Single agent stops, others continue"
```

**2. Global Pool Exhaustion**
```yaml
trigger: Total consumed + pool available = total budget
action:
  - Stop allocating from pool
  - Warn all active agents
  - Prioritize critical agents
  - Begin graceful degradation
impact: "System-wide budget pressure"
```

**3. Cascading Exhaustion**
```yaml
trigger: Multiple agents hit limits simultaneously
risk: "Workflow failure if no agents can continue"
prevention:
  - Reserve minimum per agent (5% of fair share)
  - Prevent any agent from monopolizing pool
  - Gradual degradation (throttle before stop)
```

**Mitigation Strategies:**

```go
// 1. Graduated Response
func HandleBudgetPressure() {
  global := GetGlobalStatus()

  if global.PoolPercentage > 0.90 {
    // Pool nearly exhausted
    for agent := range agents {
      Throttle(agent, rate=0.5)  // Slow everyone down 50%
    }
  }

  if global.PoolPercentage >= 1.0 {
    // Pool exhausted
    nonCriticalAgents := agents.Filter(a => a.Priority != "critical")
    for agent := range nonCriticalAgents {
      Stop(agent, "budget_exhausted")
    }
  }
}

// 2. Emergency Reserve
reserved := totalBudget * 0.10  // 10% emergency reserve
pool := totalBudget - reserved

if global.Crisis() {
  // Release emergency reserve
  pool += reserved
}
```

### Key Insight
**Progressive degradation prevents abrupt failures.** Throttle → Degrade → Stop critical, then all.

---

## Iteration 7: Final Synthesis - Optimal Resource Manager

### Synthesis

**RESOURCE_MANAGER_AGENT** is an **L4_ADAPTIVE** agent that centralizes budget lifecycle management.

**Core Architecture:**

```
┌───────────────────────────────────────┐
│       RESOURCE MANAGER AGENT          │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │  Allocation Engine              │ │
│  │  (Equal, Proportional, Pool)    │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Consumption Tracker            │ │
│  │  (Aggregate Usage Reports)      │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Enforcement Engine             │ │
│  │  (Green→Yellow→Orange→Red→Black)│ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Forecasting & Reporting        │ │
│  │  (Predict, Analyze, Recommend)  │ │
│  └──────────────┬──────────────────┘ │
│                 │                     │
│  ┌──────────────▼──────────────────┐ │
│  │  Reallocation Engine            │ │
│  │  (Dynamic Pool Management)      │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
```

**Quality Metrics:**
```yaml
utilization:
  target: ≥0.85
  measurement: "consumed / allocated"
  current: 0.88
  note: "Dynamic reallocation improves from static ~60%"

fairness:
  target: ≥0.90
  measurement: "1 - variance(allocations)/mean(allocations)"
  current: 0.92

responsiveness:
  target: ≤1s
  measurement: "Time from threshold to enforcement"
  current: 0.3s

forecast_accuracy:
  target: ≥0.85
  measurement: "% of predictions within 10% of actual"
  current: 0.88
```

**When to Use:**
```
✅ Use Resource Manager for:
  - Multi-agent workflows with shared budget
  - Dynamic workloads (varying complexity)
  - Need for global visibility and control
  - Budget optimization (high utilization)

❌ Don't use Resource Manager for:
  - Single agent (no coordination needed)
  - Fixed, predictable workloads (static allocation sufficient)
  - Development/testing (overhead not worth it)
```

### Key Insight
**Resource Manager is orchestrator for budgets**, not tasks. Manages resources, not logic.

---

## Meta-Learning

What did we learn about **Resource Manager** through meta-prompting?

1. **Centralization enables global optimization** - See and control total resource usage
2. **Self-reporting reduces cognitive load** - Agents report, manager enforces
3. **Dynamic allocation improves utilization** - Rebalance based on actual needs
4. **Progressive enforcement prevents failures** - Throttle before stopping
5. **Forecasting enables proactive action** - Warn before exhaustion
6. **Pool management provides flexibility** - Borrow/return creates elasticity
7. **Resource Manager ≠ Orchestrator** - Manages budgets, not workflows

The pattern: **Allocate budgets → Collect usage reports → Enforce limits → Forecast exhaustion → Reallocate dynamically → Report efficiency**
