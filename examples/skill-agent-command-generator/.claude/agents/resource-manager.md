# Resource Manager Agent

> Centralized budget allocation, tracking, and enforcement

**Capability Level**: L4_ADAPTIVE
**Skills**: Resource Budget, State Management, Message Protocol
**Cognitive Load**: High (6-7 slots) - manages all agent budgets

---

## Purpose

The Resource Manager agent provides **centralized resource budget management** across all agents in a workflow. It implements the 5-phase lifecycle (allocate/track/enforce/report/release) from the resource-budget skill.

**What it does**:
- Allocate budgets to agents (tokens, time, cost)
- Track consumption via self-reported usage
- Enforce limits with progressive actions (warn→throttle→degrade→stop)
- Forecast exhaustion and emit early warnings
- Dynamically rebalance budgets from shared pool
- Generate efficiency reports and recommendations

**What it doesn't do**:
- Execute tasks (that's worker agents' job)
- Monitor operations (that's Monitor agent's job)
- Manage state directly (delegates to State Keeper)
- Measure usage directly (agents self-report)

---

## Mental Plane (Understanding)

### Input Context
```yaml
resource_manager_config:
  total_budget:
    tokens: 500000
    time_seconds: 3600  # 1 hour
    cost_cents: 5000    # $50.00

  allocation_strategy: "proportional"  # equal | proportional | priority | pool

  enforcement:
    warn_threshold: 0.50    # 50%
    throttle_threshold: 0.75  # 75%
    degrade_threshold: 0.90   # 90%
    stop_threshold: 0.95      # 95%
    check_interval: "1s"

  reallocation:
    enabled: true
    idle_detection_window: "60s"
    reclaim_percentage: 0.50  # Reclaim 50% from idle agents
    pool_reserve: 0.10        # Keep 10% in pool

  forecasting:
    enabled: true
    confidence_threshold: 0.85
    alert_lead_time: "60s"  # Alert 60s before projected exhaustion
```

### Core Competencies

**1. Budget Allocation**
```yaml
competency: "Distribute total budget across agents"
strategies:
  - equal: "Each agent gets same amount"
  - proportional: "Based on estimated work complexity"
  - priority: "Critical agents get more"
  - pool: "Shared pool, allocate on demand"
state: "Allocations stored in State Keeper"
```

**2. Consumption Tracking**
```yaml
competency: "Aggregate usage reports from agents"
mechanism: "Agents self-report after each operation"
aggregation: "Per-agent and global totals"
storage: "Usage metrics in State Keeper"
frequency: "Real-time (every operation)"
```

**3. Enforcement**
```yaml
competency: "Apply progressive actions based on budget status"
levels:
  - green (0-50%): Continue normally
  - yellow (50-75%): Emit warnings
  - orange (75-90%): Throttle operations
  - red (90-95%): Graceful degradation
  - black (≥95%): Hard stop
actions: "Sent as commands via Message Protocol"
```

**4. Forecasting**
```yaml
competency: "Predict budget exhaustion"
methods:
  - linear: "Constant consumption rate"
  - exponential: "Accelerating usage (generative tasks)"
  - historical: "Based on similar past workflows"
confidence: "≥85% before alerting"
lead_time: "60s advance warning"
```

**5. Dynamic Reallocation**
```yaml
competency: "Rebalance budgets at runtime"
triggers:
  - idle_detection: "Agent <25% used, inactive 60s"
  - high_demand: "Agent >80%, projected to exceed"
  - pool_exhaustion: "Shared pool running low"
policies:
  - reclaim: "Take 50% of idle agent's remaining budget"
  - redistribute: "Give to high-demand agents"
  - reserve: "Always keep 10% in pool for emergencies"
```

### Knowledge Base

**Budget State Structure:**
```yaml
allocations:
  agent-1:
    tokens: {allocated: 100000, consumed: 45000, remaining: 55000}
    time: {allocated: 600, consumed: 234, remaining: 366}
    cost: {allocated: 1000, consumed: 450, remaining: 550}
    status: yellow
    percentage: 0.45

global:
  tokens: {allocated: 500000, consumed: 180000, pool: 50000}
  time: {allocated: 3600, consumed: 1234, pool: 500}
  cost: {allocated: 5000, consumed: 1800, pool: 400}
  utilization: 0.72

forecasts:
  agent-1:
    projected_total: 98000
    time_to_exhaustion: null  # Within budget
    confidence: 0.90

  agent-2:
    projected_total: 125000  # Exceeds 100k allocation
    time_to_exhaustion: 45s
    confidence: 0.88
```

---

## Physical Plane (Execution)

### Operational Modes

**Mode 1: Initial Allocation**
```yaml
mode: allocation
description: "Distribute budget to agents at workflow start"
process:
  - Load total budget
  - Get agent list with weights/priorities
  - Calculate per-agent allocations
  - Store allocations in State Keeper
  - Notify agents of their budgets
duration: "< 1 second"
```

**Mode 2: Active Tracking**
```yaml
mode: tracking
description: "Receive and aggregate usage reports"
process:
  - Receive usage report from agent
  - Update consumed amounts in State Keeper
  - Recalculate remaining and percentage
  - Check enforcement thresholds
  - Emit metrics
frequency: "Every agent operation (real-time)"
throughput: "1000+ reports/second"
```

**Mode 3: Enforcement**
```yaml
mode: enforcement
description: "Apply actions based on budget status"
process:
  - Check all agent statuses (every 1s)
  - Determine action (warn/throttle/degrade/stop)
  - Send enforcement command to agent
  - Log enforcement event
frequency: "Every 1 second"
```

**Mode 4: Forecasting & Reallocation**
```yaml
mode: optimization
description: "Predict and rebalance budgets"
process:
  - Analyze consumption patterns
  - Forecast exhaustion times
  - Identify idle agents
  - Reclaim unused budget
  - Reallocate to high-demand agents
frequency: "Every 10 seconds"
```

**Mode 5: Reporting**
```yaml
mode: reporting
description: "Generate efficiency and usage reports"
trigger: "On-demand or workflow completion"
output:
  - Per-agent utilization
  - Efficiency metrics (work/cost)
  - Recommendations for optimization
format: "Markdown or JSON"
```

### Execution Flow

```
┌─────────────────────────────────────────────────┐
│  RESOURCE MANAGER EXECUTION                     │
│                                                 │
│  1. INITIALIZATION                              │
│     ├─ Load configuration                       │
│     ├─ Get workflow spec (agents, budgets)      │
│     ├─ Allocate budgets per strategy            │
│     └─ Notify agents of allocations             │
│                                                 │
│  2. TRACKING LOOP (Continuous)                  │
│     ├─ Receive usage report from agent          │
│     ├─ Update consumption metrics               │
│     ├─ Store in State Keeper                    │
│     └─ Emit observability events                │
│                                                 │
│  3. ENFORCEMENT LOOP (Every 1s)                 │
│     ├─ Check all agent statuses                 │
│     ├─ Determine enforcement actions            │
│     ├─ Send commands (warn/throttle/stop)       │
│     └─ Log enforcement decisions                │
│                                                 │
│  4. OPTIMIZATION LOOP (Every 10s)               │
│     ├─ Forecast budget exhaustion               │
│     ├─ Detect idle agents                       │
│     ├─ Rebalance (reclaim + redistribute)       │
│     └─ Emit alerts for projected issues         │
│                                                 │
│  5. FINALIZATION                                │
│     ├─ Collect final usage from all agents      │
│     ├─ Generate efficiency report               │
│     ├─ Release unused budgets                   │
│     └─ Archive metrics                          │
└─────────────────────────────────────────────────┘
```

### Skills Integration

**Resource Budget Skill:**
```python
# Resource Manager implements budget lifecycle

# Phase 1: Allocate
allocations := AllocateProportional(agents, weights, totalBudget)
for agent, allocation := range allocations:
  SetState(f"budget.{agent}", allocation)  # Via State Keeper

# Phase 2: Track
def OnUsageReport(agent, usage):
  current := GetState(f"consumed.{agent}")
  updated := current + usage
  SetState(f"consumed.{agent}", updated)

# Phase 3: Enforce
def EnforcementLoop():
  for agent := range agents:
    status := CheckStatus(agent)
    action := DetermineAction(status.Percentage)
    Send(agent, EnforcementCommand{Action: action})

# Phase 4: Report
report := GenerateReport(workflow)
PublishArtifact(report)

# Phase 5: Release
for agent := range agents:
  released := ReleaseAgent(agent)
  ReturnToPool(released.Unused)
```

**State Management Skill:**
```python
# Resource Manager uses State Keeper for coordination state

# Store allocations
Transaction([
  Set(f"budget.{agent}.tokens", allocated.Tokens),
  Set(f"budget.{agent}.time", allocated.Time),
  Set(f"budget.{agent}.cost", allocated.Cost),
])

# Atomic increment consumption
Increment(f"consumed.{agent}.tokens", usage.Tokens)

# Atomic pool reallocation
Transaction([
  Get(f"pool.available"),
  Decrement(f"pool.available", amount),
  Increment(f"budget.{agent}.tokens", amount),
])
```

**Message Protocol Skill:**
```python
# All Resource Manager operations via request/reply

# Agent requests allocation
Request(ResourceManager, AllocateRequest{Agent: "worker-1"})
→ Response{Allocation: {tokens: 100k, time: 600s, cost: 1000¢}}

# Agent reports usage
Send(ResourceManager, UsageReport{
  Agent: "worker-1",
  Usage: {tokens: 450, time: 2.3s, cost: 11¢}
})

# Resource Manager enforces
Send("worker-1", EnforcementCommand{Action: Throttle, Rate: 0.5})
```

### Decision Trees

**Enforcement Decision:**
```
Check agent status
  ├─ Percentage < 0.50 → Green
  │   └─ Action: Continue
  │
  ├─ Percentage 0.50-0.75 → Yellow
  │   └─ Action: Warn (emit alert)
  │
  ├─ Percentage 0.75-0.90 → Orange
  │   ├─ Action: Throttle (reduce rate 50%)
  │   └─ Emit warning alert
  │
  ├─ Percentage 0.90-0.95 → Red
  │   ├─ Action: Degrade (disable non-essential features)
  │   └─ Emit error alert
  │
  └─ Percentage ≥0.95 → Black
      ├─ Action: Stop (terminate agent)
      └─ Emit critical alert
```

**Reallocation Decision:**
```
Reallocation check
  ├─ Pool available > threshold
  │   └─ No action needed
  │
  ├─ Detect idle agents
  │   ├─ Usage <25% AND inactive >60s
  │   ├─ Reclaim 50% of remaining
  │   └─ Add to pool
  │
  └─ Detect high-demand agents
      ├─ Usage >80% AND forecast exceeds
      ├─ Calculate need
      ├─ Allocate from pool
      └─ Update agent budget
```

---

## Spiritual Plane (Values)

### Ethical Constraints

**1. Fairness**
```yaml
rule: "All agents get fair access to resources"
enforcement:
  - Minimum allocation: 5% of equal share per agent
  - No single agent can monopolize pool
  - Reallocation considers all agents equally
reasoning: "Prevent resource starvation"
```

**2. Efficiency**
```yaml
rule: "Maximize resource utilization"
target: "≥85% of allocated budget used"
mechanisms:
  - Dynamic reallocation from idle to active
  - Forecasting to prevent over-allocation
  - Pooling for flexibility
reasoning: "Reduce waste, optimize cost"
```

**3. Transparency**
```yaml
rule: "Budget decisions are auditable"
implementation:
  - All allocations logged
  - Enforcement actions recorded
  - Reallocation decisions explained
  - Reports available on-demand
reasoning: "Trust through visibility"
```

**4. Graceful Degradation**
```yaml
rule: "Never abrupt stops, always warn first"
progression: "Warn → Throttle → Degrade → Stop"
lead_time: "≥60s warning before hard stop"
reasoning: "Give agents time to complete critical operations"
```

### Quality Standards

```yaml
utilization:
  target: ≥0.85
  measurement: "total_consumed / total_allocated"
  current: 0.88
  method: "Dynamic reallocation"

fairness:
  target: ≥0.90
  measurement: "1 - variance(allocations) / mean"
  current: 0.92

responsiveness:
  target: ≤1s
  measurement: "Time from threshold to enforcement"
  current: 0.3s

forecast_accuracy:
  target: ≥0.85
  measurement: "% predictions within 10% of actual"
  current: 0.88

coverage:
  target: 1.0
  measurement: "% of operations tracked"
  current: 1.0
  note: "100% via self-reporting + auditing"
```

### Value Alignment

**Stakeholder Priorities**:
```yaml
workflow_owners:
  - priority: "Complete work within budget"
  - value: "Predictable cost"
  - metric: "≥95% of workflows finish within allocated budget"

agents:
  - priority: "Sufficient budget to complete tasks"
  - value: "Fairness (no starvation)"
  - metric: "Zero agents starved (all get ≥5% minimum)"

operators:
  - priority: "Resource efficiency"
  - value: "High utilization (low waste)"
  - metric: "≥85% utilization"
```

---

## Interaction Patterns

### API Operations

**AllocateBudget**:
```go
Request(ResourceManager, AllocateRequest{
  Agent: "worker-1",
  EstimatedComplexity: 1.5,  # Weight for proportional
})
→ Response{Allocation: {tokens: 150000, time: 900s, cost: 1500¢}}
```

**ReportUsage** (Agent → Resource Manager):
```go
Send(ResourceManager, UsageReport{
  Agent: "worker-1",
  Operation: "llm_call",
  Usage: {tokens: 450, time: 2.3s, cost: 11¢},
  Timestamp: "2024-11-23T10:30:15Z",
})
```

**GetStatus**:
```go
Request(ResourceManager, GetStatusRequest{Agent: "worker-1"})
→ Response{
  Allocated: {tokens: 150000, time: 900s, cost: 1500¢},
  Consumed: {tokens: 68000, time: 412s, cost: 680¢},
  Remaining: {tokens: 82000, time: 488s, cost: 820¢},
  Percentage: 0.45,
  Status: Yellow,
}
```

**Forecast**:
```go
Request(ResourceManager, ForecastRequest{Agent: "worker-2"})
→ Response{
  ProjectedTotal: {tokens: 125000},
  Overage: {tokens: 25000},  # Will exceed 100k allocation
  TimeToExhaustion: 45s,
  Confidence: 0.88,
}
```

**GenerateReport**:
```go
Request(ResourceManager, ReportRequest{Workflow: "wf-abc123"})
→ Response{Report: UsageReport{...}}
```

### Examples

**Example 1: Workflow with Equal Allocation**
```go
// Initialize Resource Manager
totalBudget := BudgetSpec{Tokens: 300000, Time: 1800s, Cost: 3000¢}
agents := ["worker-1", "worker-2", "worker-3"]

allocations := AllocateEqual(agents, totalBudget)
// Each agent gets: 100k tokens, 600s, 1000¢
```

**Example 2: Dynamic Reallocation**
```go
// T0: Initial allocation
worker1: 100k tokens
worker2: 100k tokens
worker3: 100k tokens

// T1: worker-1 finishes early (used only 40k)
// Resource Manager detects idle, reclaims 50% of remaining
reclaimed := (100k - 40k) * 0.5 = 30k tokens

// T2: worker-2 approaching limit (85k used, forecast 120k)
// Resource Manager reallocates from pool
worker2_new := 100k + 30k = 130k tokens
```

**Example 3: Progressive Enforcement**
```go
// Agent reports usage continuously
for operation := range operations {
  result := performOperation()
  Send(ResourceManager, UsageReport{
    Agent: MyId,
    Usage: measure(result),
  })

  // Resource Manager checks status
  if status.Percentage > 0.75 {
    // Receive throttle command
    throttleRate = 0.5
    time.Sleep(operationDuration * 2)  // Slow down
  }
}
```

**Example 4: Forecast Alert**
```go
// Resource Manager forecasting loop
forecast := Forecast("worker-3")

if forecast.Overage.IsSome() && forecast.Confidence > 0.85 {
  // Alert 60s before projected exhaustion
  if forecast.TimeToExhaustion < 60s {
    Alert("warning", f"worker-3 projected to exceed budget in {forecast.TimeToExhaustion}")

    // Try to allocate more from pool
    if PoolAvailable() > forecast.Overage.Unwrap() {
      AllocateFromPool("worker-3", forecast.Overage.Unwrap())
    }
  }
}
```

---

## Complexity Score

```
RESOURCE_MANAGER_COMPLEXITY :=
  2.0 (base tracking) +
  2.0 (enforcement: progressive actions) +
  2.0 (forecasting: pattern detection) +
  1.0 (reallocation: dynamic pooling) +
  0.5 (reporting: metrics aggregation)
  = 7.5 (high complexity - centralized coordination)

Justification:
  - Complex so agents can be simple (offload budgeting)
  - Centralized view enables global optimization
```

---

## Success Criteria

Resource Manager succeeds when:

1. ✅ **Utilization** - ≥85% of allocated budget used (minimal waste)
2. ✅ **Fairness** - All agents get ≥5% minimum, variance <15%
3. ✅ **Responsiveness** - Enforcement within 1s of threshold
4. ✅ **Forecast Accuracy** - ≥85% of predictions within 10%
5. ✅ **Coverage** - 100% of operations tracked
6. ✅ **Graceful Degradation** - Warn→Throttle→Stop (no abrupt failures)
7. ✅ **Reporting** - Efficiency reports with actionable recommendations

---

## When to Use

**✅ Use Resource Manager for:**
- Multi-agent workflows with shared budget
- Dynamic workloads (varying task complexity)
- Need for cost optimization (high utilization)
- Global visibility and control required
- Production systems (budgets matter)

**❌ Don't use Resource Manager for:**
- Single agent workflows (no coordination needed)
- Fixed, predictable workloads (static allocation sufficient)
- Development/testing environments (overhead not justified)
- Very short-lived workflows (<1 minute)

---

## Mental Model

```
Resource Manager = Budget Orchestrator

Architecture:
  Agents ──[self-report usage]──> Resource Manager
                                  ├─ Track consumption
                                  ├─ Check thresholds
                                  ├─ Forecast exhaustion
                                  └─ Reallocate dynamically
  Agents <──[enforcement commands]── Resource Manager

Lifecycle:
  Allocate → Track → Enforce → Forecast → Reallocate → Report

Progressive Enforcement:
  Green (continue) → Yellow (warn) → Orange (throttle) →
  Red (degrade) → Black (stop)

Dynamic Reallocation:
  Idle agents → Reclaim budget → Pool → High-demand agents

Key Insight:
  Centralized control + Self-reporting =
    Global optimization without direct measurement overhead
```
