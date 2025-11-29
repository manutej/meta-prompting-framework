# /monitor - Real-time Workflow Observability

Monitor running workflows with live metrics, logs, and alerts.

## Usage

```bash
/monitor <workflow_id> [options]
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `workflow_id` | string | Yes | ID of workflow to monitor |
| `--mode` | enum | No | Output mode: `dashboard`, `stream`, `report`, `trace` (default: `dashboard`) |
| `--agents` | string[] | No | Filter to specific agent IDs (default: all) |
| `--metrics` | string[] | No | Which metrics to track (default: `tokens,time,cost,errors`) |
| `--sample-rate` | float | No | Sampling percentage 0.0-1.0 (default: `0.1` for 10%) |
| `--alert` | string[] | No | Alert rules in format `metric>threshold` |
| `--refresh` | duration | No | Dashboard refresh interval (default: `1s`) |
| `--follow` | bool | No | Stream mode: follow like `tail -f` (default: `false`) |
| `--window` | duration | No | Time window for metrics (default: `60s`) |
| `--format` | enum | No | Output format: `ascii`, `json`, `markdown` (default: `ascii`) |

## Options Detail

### --mode

**dashboard** (default)
- Live-updating display of workflow state
- Refreshes every `--refresh` interval
- Shows agents, resources, metrics, recent errors
- Interactive: Press 'q' to quit, 'r' to refresh

**stream**
- Continuous event feed (like `tail -f`)
- Emits events as they occur
- Can pipe to external tools
- Use `--follow` to keep connection open

**report**
- Generate summary report once
- Aggregated statistics over time window
- Recommendations and anomaly detection
- Suitable for archiving or sharing

**trace**
- Visualize distributed execution traces
- Shows parent-child span relationships
- Identifies bottlenecks and critical paths
- Filter by agent, request ID, or error status

### --alert

Define alert conditions:

```bash
# Alert when error rate exceeds 5%
/monitor wf-123 --alert="error_rate>0.05"

# Multiple alerts
/monitor wf-123 \
  --alert="error_rate>0.05" \
  --alert="budget.percentage>0.75" \
  --alert="latency.p95>10s"
```

Alert condition syntax:
```
<metric> <operator> <threshold>

Operators: >, <, >=, <=, ==, !=
Examples:
  - "tokens>50000"
  - "duration>=300s"
  - "cost_cents>1000"
  - "active_agents<1"
```

### --format

**ascii**: Human-readable tables and boxes (default for dashboard)
**json**: Structured JSON output (default for stream)
**markdown**: Formatted markdown (default for report)

## Examples

### Example 1: Live Dashboard

```bash
/monitor wf-abc123
```

**Output**:
```
╔═══════════════════════════════════════════════════════════╗
║  MONITORING DASHBOARD - wf-abc123                         ║
╠═══════════════════════════════════════════════════════════╣
║  Workflow Status: RUNNING                                 ║
║  Duration: 00:02:34                                       ║
║  Agents: 3 active, 0 idle, 0 failed                       ║
╠═══════════════════════════════════════════════════════════╣
║  RESOURCE USAGE                                           ║
║  ┌─────────────┬────────┬─────────┬─────────┬──────────┐ ║
║  │ Agent       │ Tokens │ Time    │ Cost    │ Budget % │ ║
║  ├─────────────┼────────┼─────────┼─────────┼──────────┤ ║
║  │ worker-1    │ 12,450 │ 00:01:23│ $0.31   │ 62% 🟡   │ ║
║  │ worker-2    │  8,230 │ 00:00:54│ $0.21   │ 41% 🟢   │ ║
║  │ worker-3    │ 15,890 │ 00:02:10│ $0.40   │ 79% 🟠   │ ║
║  │ TOTAL       │ 36,570 │ 00:02:34│ $0.92   │ 61%      │ ║
║  └─────────────┴────────┴─────────┴─────────┴──────────┘ ║
╠═══════════════════════════════════════════════════════════╣
║  METRICS (Last 60s)                                       ║
║  • Tasks completed: 47 (0.78/s)                           ║
║  • Error rate: 2.1% (below threshold)                     ║
║  • Avg latency: P50=2.3s P95=7.1s P99=12.4s               ║
║  • Token efficiency: 778 tokens/task                      ║
╠═══════════════════════════════════════════════════════════╣
║  RECENT ERRORS (Last 3)                                   ║
║  [00:02:12] worker-3: Budget warning (75% consumed)       ║
║  [00:01:45] worker-1: Retry attempt 2/3 for task-42       ║
║  [00:00:31] worker-2: Temporary API timeout (recovered)   ║
╠═══════════════════════════════════════════════════════════╣
║  ALERTS: 1 warning, 0 errors                              ║
╚═══════════════════════════════════════════════════════════╝

Refreshing every 1s... (Press 'q' to quit, 'r' to refresh)
```

---

### Example 2: Event Stream with Filtering

```bash
/monitor wf-abc123 --mode=stream --agents=worker-1 --follow
```

**Output**:
```json
{"timestamp":"2024-11-23T10:30:01Z","agent":"worker-1","type":"task.started","task_id":"task-42"}
{"timestamp":"2024-11-23T10:30:03Z","agent":"worker-1","type":"llm.call","tokens":450,"duration_ms":1200}
{"timestamp":"2024-11-23T10:30:04Z","agent":"worker-1","type":"task.completed","task_id":"task-42","tokens":450}
{"timestamp":"2024-11-23T10:30:05Z","agent":"worker-1","type":"task.started","task_id":"task-43"}
...
```

Stream can be piped to external tools:
```bash
/monitor wf-abc123 --mode=stream --follow | jq '.tokens' | stats
```

---

### Example 3: Summary Report

```bash
/monitor wf-abc123 --mode=report --window=5m
```

**Output**:
```markdown
# Monitoring Report: wf-abc123

## Executive Summary
- **Duration**: 00:05:12
- **Status**: COMPLETED ✅
- **Agents**: 3 agents (all succeeded)
- **Tasks**: 127 completed (0.41/s)
- **Cost**: $1.35 (within budget)
- **Health**: ✅ HEALTHY

## Resource Consumption

| Metric | Used    | Allocated | Utilization | Efficiency |
|--------|---------|-----------|-------------|------------|
| Tokens | 98,450  | 150,000   | 66%         | 775/task   |
| Time   | 312s    | 600s      | 52%         | 2.5s/task  |
| Cost   | $1.35   | $2.00     | 68%         | $0.011/task|

**Assessment**: Good utilization. Finished under time and budget.

## Agent Performance

### worker-1
- **Tasks**: 45 (35%)
- **Tokens**: 34,890 (775/task)
- **Duration**: 112s (2.5s/task)
- **Errors**: 1 (2.2%)
- **Rating**: ⭐⭐⭐⭐ Excellent

### worker-2
- **Tasks**: 38 (30%)
- **Tokens**: 29,260 (770/task)
- **Duration**: 96s (2.5s/task)
- **Errors**: 0 (0%)
- **Rating**: ⭐⭐⭐⭐⭐ Perfect

### worker-3
- **Tasks**: 44 (35%)
- **Tokens**: 34,300 (780/task)
- **Duration**: 104s (2.4s/task)
- **Errors**: 2 (4.5%)
- **Rating**: ⭐⭐⭐⭐ Very Good

## Anomalies Detected

1. **Temporary latency spike at 00:02:15** (Severity: INFO)
   - P99 latency reached 15.2s (vs baseline 8.0s)
   - Duration: 30s
   - Cause: External API slowdown
   - Resolution: Automatically recovered
   - Impact: Minimal (2 tasks affected)

2. **worker-3 budget warning at 00:03:45** (Severity: WARNING)
   - Reached 78% budget consumption
   - Projected to finish within budget (✅ confirmed)
   - No action taken

## Recommendations

✅ **Performance**: All agents performed within expected parameters
✅ **Efficiency**: Token usage consistent with baseline (775/task)
✅ **Reliability**: 97.6% success rate (above 95% target)
💡 **Future**: Consider allocating more budget to worker-3 for complex tasks

---

**Report generated**: 2024-11-23T10:35:12Z
**Monitoring overhead**: 2.3% (within 5% target)
```

---

### Example 4: Budget Alert

```bash
/monitor wf-abc123 \
  --alert="budget.percentage>0.75" \
  --alert="budget.percentage>0.90"
```

**Output**:
```
[00:01:34] ⚠️  WARNING: worker-3 budget.percentage (0.76) > 0.75
  Agent: worker-3
  Current: 76% of budget consumed
  Remaining: 12,000 tokens, 72s, $0.30
  Projection: Will finish at 88% budget (within limits)
  Recommendation: Monitor closely, no action needed yet

[00:02:45] 🚨 ERROR: worker-1 budget.percentage (0.91) > 0.90
  Agent: worker-1
  Current: 91% of budget consumed
  Remaining: 4,500 tokens, 27s, $0.11
  Projection: Will exceed budget by 5% if trend continues
  Recommendation: Consider throttling or degrading quality
```

---

### Example 5: Trace Visualization

```bash
/monitor wf-abc123 --mode=trace --filter="status=error"
```

**Output**:
```
Trace: req-789 (FAILED)
Duration: 8.2s
Status: ERROR
Error: Task timeout after 8.0s

process_workflow [0ms────────────────────────8200ms] (8.2s)
├─ spawn_worker-1 [100ms──200ms] (100ms)
├─ spawn_worker-2 [150ms──250ms] (100ms)
├─ spawn_worker-3 [180ms──280ms] (100ms)
│
├─ worker-1: process_task [200ms──────2500ms] (2.3s) ✅
│  ├─ load_context [210ms──310ms] (100ms)
│  ├─ llm_call [320ms────1800ms] (1.48s)
│  └─ store_result [1810ms──2500ms] (690ms)
│
├─ worker-2: process_task [250ms──────2800ms] (2.55s) ✅
│  ├─ load_context [260ms──360ms] (100ms)
│  ├─ llm_call [370ms────1900ms] (1.53s)
│  └─ store_result [1910ms──2800ms] (890ms)
│
└─ worker-3: process_task [280ms────────────8200ms] (7.92s) ❌
   ├─ load_context [290ms──390ms] (100ms)
   ├─ llm_call [400ms──────────7900ms] (7.5s) ⚠️  SLOW
   └─ timeout [8000ms] ❌

Critical Path: process_workflow → worker-3 → llm_call (7.5s)
Bottleneck: worker-3.llm_call (91% of total time)

Recommendation: Investigate why worker-3 LLM call took 7.5s
```

---

### Example 6: Focus on Specific Agent

```bash
/monitor wf-abc123 --agents=worker-2 --metrics=tokens,errors
```

**Output**:
```
╔═══════════════════════════════════════════════════════════╗
║  AGENT MONITOR - worker-2                                 ║
╠═══════════════════════════════════════════════════════════╣
║  Status: ACTIVE                                           ║
║  Tasks: 23 completed, 1 in progress                       ║
║  Duration: 00:01:42                                       ║
╠═══════════════════════════════════════════════════════════╣
║  TOKEN USAGE                                              ║
║  Current:  18,450 tokens                                  ║
║  Budget:   50,000 tokens (37% used)                       ║
║  Rate:     180 tokens/s                                   ║
║  Avg/task: 803 tokens/task                                ║
║  Projected: ~38,000 tokens total (within budget)          ║
╠═══════════════════════════════════════════════════════════╣
║  ERRORS                                                   ║
║  Total: 0 errors (0%)                                     ║
║  Status: ✅ Perfect record                                ║
╠═══════════════════════════════════════════════════════════╣
║  RECENT ACTIVITY                                          ║
║  [00:01:40] task-45 completed (850 tokens)                ║
║  [00:01:35] task-44 completed (790 tokens)                ║
║  [00:01:30] task-43 completed (765 tokens)                ║
║  [00:01:42] task-46 started                               ║
╚═══════════════════════════════════════════════════════════╝
```

---

### Example 7: JSON Export for Analysis

```bash
/monitor wf-abc123 --mode=report --format=json > report.json
```

**Output** (report.json):
```json
{
  "workflow_id": "wf-abc123",
  "duration_seconds": 312,
  "status": "COMPLETED",
  "agents": {
    "total": 3,
    "active": 0,
    "completed": 3,
    "failed": 0
  },
  "resources": {
    "tokens": {
      "used": 98450,
      "allocated": 150000,
      "utilization": 0.66,
      "efficiency": 775
    },
    "time": {
      "used": 312,
      "allocated": 600,
      "utilization": 0.52
    },
    "cost": {
      "used_cents": 135,
      "allocated_cents": 200,
      "utilization": 0.68
    }
  },
  "metrics": {
    "tasks_completed": 127,
    "tasks_per_second": 0.41,
    "error_rate": 0.024,
    "latency": {
      "p50": 2.3,
      "p95": 7.1,
      "p99": 12.4
    }
  },
  "agents_detail": [
    {
      "agent_id": "worker-1",
      "tasks": 45,
      "tokens": 34890,
      "duration_seconds": 112,
      "errors": 1,
      "error_rate": 0.022,
      "rating": 4
    },
    ...
  ],
  "anomalies": [
    {
      "timestamp": "2024-11-23T10:32:15Z",
      "type": "latency_spike",
      "severity": "info",
      "description": "Temporary latency spike",
      "impact": "minimal"
    }
  ],
  "recommendations": [
    "Performance: All agents within expected parameters",
    "Consider allocating more budget to worker-3"
  ]
}
```

---

### Example 8: Continuous Monitoring with Multiple Alerts

```bash
/monitor wf-abc123 \
  --mode=dashboard \
  --alert="error_rate>0.05" \
  --alert="tokens>100000" \
  --alert="duration>=600s" \
  --alert="budget.percentage>0.90" \
  --refresh=2s
```

Dashboard updates every 2s, alerts trigger when conditions met.

---

## Output Modes Summary

| Mode | Use Case | Format | Real-time | Verbosity |
|------|----------|--------|-----------|-----------|
| `dashboard` | Live monitoring | ASCII table | Yes | Medium |
| `stream` | Pipe to tools | JSON lines | Yes | High |
| `report` | Summary analysis | Markdown | No | Low |
| `trace` | Debug failures | ASCII tree | No | Very High |

---

## Implementation

When `/monitor` is invoked:

1. **Spawn Monitor Agent** with monitoring_spec
2. **Subscribe to workflow events** from all specified agents
3. **Collect metrics** according to sample rate
4. **Analyze and aggregate** per time window
5. **Evaluate alert rules** each refresh cycle
6. **Render output** in specified mode and format
7. **Continue until**:
   - Workflow completes (report mode)
   - User quits ('q' in dashboard)
   - Ctrl+C signal (stream mode)

---

## Notes

- **Overhead**: Monitor agent typically consumes 2-5% of workflow resources
- **Sampling**: Default 10% captures enough signal while minimizing cost
- **Retention**: Raw events kept for 10 minutes, aggregates for 1 hour
- **Accuracy**: Metrics typically within 5% of ground truth
- **Latency**: Dashboard updates within 1 second of events

---

## Error Handling

If workflow doesn't exist:
```
Error: Workflow 'wf-invalid' not found
Available workflows: wf-abc123, wf-def456
```

If monitor overhead exceeds limit:
```
Warning: Monitoring overhead at 6.2% (target <5%)
Automatically reducing sample rate to 5%
```

If invalid alert syntax:
```
Error: Invalid alert condition 'error_rate>>0.05'
Expected format: <metric> <operator> <threshold>
Operators: >, <, >=, <=, ==, !=
```

---

## Related Commands

- `/orchestrate` - Spawn and coordinate multiple agents
- `/spawn` - Launch single agent
- `/budget` - Manage resource allocations
- `/state` - Query workflow state

---

## Advanced Usage

### Pipe stream to external analyzer

```bash
/monitor wf-abc123 --mode=stream --follow \
  | jq 'select(.type=="error")' \
  | analyze-errors.py
```

### Monitor multiple workflows

```bash
# Terminal 1
/monitor wf-abc123 --mode=dashboard

# Terminal 2
/monitor wf-def456 --mode=dashboard
```

### Generate historical report

```bash
# After workflow completes
/monitor wf-abc123 --mode=report --window=1h --format=markdown
```
