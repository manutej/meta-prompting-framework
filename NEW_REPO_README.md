# Agentic Skill Architecture

> Production-ready multi-agent system built through 7-iteration meta-prompting

A comprehensive framework for building sophisticated multi-agent systems with hierarchical skills (L1-L7), specialized agents, and full orchestration capabilities. Features state management, resource budgeting, real-time monitoring, and formal message passing—all generated through rigorous 7-iteration meta-prompting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-skill-architecture
cd agentic-skill-architecture

# The .claude/ configuration is ready to use
# Drop src/.claude/ into your project to use skills, agents, and commands

# Example: Use the orchestrator to run a multi-agent workflow
/orchestrate "process 100 tasks" --pattern=parallel --max-agents=5
```

---

## 🎯 What Is This?

This repository contains a **complete multi-agent architecture** with:

- **12 Skills** (L1-L7 hierarchy + 5 agentic skills)
- **7 Specialized Agents** (orchestration, monitoring, state, budgeting, quality, composition, evolution)
- **7 User Commands** (generate, compose, validate, evolve, orchestrate, spawn, monitor)
- **7-Iteration Meta-Prompting Workflows** (showing how everything was built)

Everything was created through **meta-prompting**—iterative refinement where each component went through 7 rounds of construct → deconstruct → reconstruct.

---

## 📚 Core Concepts

### Hierarchical Skills (L1-L7)

Skills progress from simple to emergent:

| Level | Skill | Cognitive Load | Purpose |
|-------|-------|----------------|---------|
| **L1** | Option Type | 0.5 slots | Null safety |
| **L2** | Result Type | 1 slot | Error handling |
| **L3** | Pipeline | 2 slots | Data transformation |
| **L4** | Effect Isolation | 3 slots | Side effect management |
| **L5** | Context Reader | 4 slots | Dependency injection |
| **L6** | Lazy Stream | 5 slots | Infinite sequences |
| **L7** | Meta Generator | 6-7 slots | Self-generation |

**Agentic Skills** (extend the hierarchy):
- **Agent Coordination**: Multi-agent patterns (sequential, parallel, hierarchical, race, swarm)
- **Agent Spawning**: Lifecycle management (define, validate, spawn, terminate)
- **State Management**: 4-layer state (context, local, artifacts, coordination)
- **Resource Budget**: 5-phase lifecycle (allocate, track, enforce, report, release)
- **Message Protocol**: 6 communication patterns (point-to-point, request/reply, pub/sub, broadcast, pipeline, scatter-gather)

### Specialized Agents

Each agent has a specific responsibility:

| Agent | Level | Purpose |
|-------|-------|---------|
| **Orchestrator** | L5_META | Multi-agent workflow coordination |
| **Monitor** | L3_PLANNING | Real-time observability (metrics, logs, traces) |
| **State Keeper** | L4_ADAPTIVE | Centralized coordination state (locks, transactions) |
| **Resource Manager** | L4_ADAPTIVE | Budget allocation and enforcement |
| **Quality Guard** | L3_PLANNING | Artifact validation |
| **Skill Composer** | L4_ADAPTIVE | Skill composition |
| **Evolution Engine** | L4_ADAPTIVE | System improvement |

### User Commands

Simple commands for complex operations:

```bash
# Generate new artifacts
/generate "rate limiting capability" --type=skill --level=4

# Compose skills
/compose L1-option-type L2-result-type --name=safe-value

# Validate quality
/validate .claude/skills/ --threshold=0.80

# Orchestrate multi-agent workflows
/orchestrate "analyze 1000 logs for errors" --pattern=parallel --max-agents=10

# Monitor running workflows
/monitor wf-abc123 --mode=dashboard --refresh=1s

# Spawn single agent
/spawn worker "process task-123"

# Evolve system based on usage
/evolve analyze --threshold=0.90
```

---

## 🏗️ Architecture

### Three-Tier Structure

```
┌─────────────────────────────────────────────┐
│  USER COMMANDS (7 commands)                 │
│  /generate /compose /validate /evolve       │
│  /orchestrate /spawn /monitor               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  AGENTS (7 specialized agents)              │
│  Orchestrator, Monitor, State Keeper,       │
│  Resource Manager, Quality Guard,           │
│  Skill Composer, Evolution Engine           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  SKILLS (12 skills: L1-L7 + 5 agentic)      │
│  Foundation: Option, Result, Pipeline...    │
│  Agentic: Coordination, State, Budget...    │
└─────────────────────────────────────────────┘
```

### Key Capabilities

**1. Multi-Agent Coordination**
- 5 patterns: Sequential, Parallel, Hierarchical, Race, Swarm
- Automatic failover and retries
- Circuit breaker resilience

**2. State Management**
- 4 layers: Context (immutable), Local (private), Artifacts (append-only), Coordination (transactional)
- Strong consistency guarantees
- Distributed locks and transactions

**3. Resource Budgeting**
- Token/time/cost tracking
- Progressive enforcement (warn → throttle → degrade → stop)
- Dynamic reallocation from shared pool
- 85%+ utilization efficiency

**4. Real-Time Monitoring**
- Three pillars: Metrics, Logs, Traces
- Live dashboards, event streams, reports
- Anomaly detection with <5% false positives

**5. Message Passing**
- 6 patterns from fire-and-forget to scatter-gather
- At-least-once delivery with deduplication
- Dead letter queue for failures

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation and first steps |
| [Architecture](docs/ARCHITECTURE.md) | System design deep-dive |
| [Skill Hierarchy](docs/SKILL_HIERARCHY.md) | L1-L7 skill progression |
| [Agent Catalog](docs/AGENT_CATALOG.md) | All agents with examples |
| [Command Reference](docs/COMMAND_REFERENCE.md) | All commands with usage |
| [Meta-Prompting Guide](docs/META_PROMPTING_GUIDE.md) | How to use 7-iteration method |
| [Integration Guide](docs/INTEGRATION_GUIDE.md) | Component integration patterns |

---

## 🔬 Meta-Prompting Methodology

Every component in this system was created using **7-iteration meta-prompting**:

### The Process

```
Iteration 1: FOUNDATION
  What is this? Why does it exist?

Iteration 2: PATTERN EXTRACTION
  What patterns emerge? How do they compose?

Iteration 3: COGNITIVE LOAD
  How does this affect working memory (7±2 slots)?

Iteration 4: FORMAL GRAMMAR
  What is the minimal algebra/API?

Iteration 5: TEMPORAL DYNAMICS
  How does this evolve over time?

Iteration 6: FAILURE MODES
  What can go wrong? How do we handle it?

Iteration 7: FINAL SYNTHESIS
  Optimal architecture with quality metrics
```

### Reproducibility

All 7-iteration workflows are in [`iterations/`](iterations/):
- [Agentic Architectures](iterations/agentic-architectures/7-ITERATION-WORKFLOW.md)
- [State Management](iterations/state-management/7-ITERATION-WORKFLOW.md)
- [Resource Budget](iterations/resource-budget/7-ITERATION-WORKFLOW.md)
- [Monitoring](iterations/monitoring/7-ITERATION-WORKFLOW.md)
- [Message Protocol](iterations/message-protocol/7-ITERATION-WORKFLOW.md)
- [State Keeper](iterations/state-keeper/7-ITERATION-WORKFLOW.md)
- [Resource Manager](iterations/resource-manager/7-ITERATION-WORKFLOW.md)

You can use these workflows as templates to create new skills, agents, or commands.

---

## 🎓 Examples

### Example 1: Multi-Agent Workflow

```bash
# Orchestrate 3 agents to process tasks in parallel
/orchestrate "process 100 tasks" --pattern=parallel --max-agents=3

# Monitor in real-time
/monitor wf-abc123 --mode=dashboard
```

**Output**:
```
╔═══════════════════════════════════════════════════════════╗
║  MONITORING DASHBOARD - wf-abc123                         ║
╠═══════════════════════════════════════════════════════════╣
║  Workflow Status: RUNNING                                 ║
║  Agents: 3 active                                         ║
║  Tasks: 67/100 completed (67%)                            ║
║  Budget: 45% consumed                                     ║
╚═══════════════════════════════════════════════════════════╝
```

### Example 2: Generate Custom Skill

```bash
# Generate a new skill using the meta-prompt generator
/generate "rate limiting capability" --type=skill --level=4

# Validate quality
/validate .claude/skills/rate-limiter.md

# Compose with existing skill
/compose rate-limiter cache-manager --name=cached-rate-limiter
```

### Example 3: Resource Management

```bash
# Allocate budget to workflow
workflow_budget = {
  tokens: 500000,
  time: 3600s,
  cost: $50
}

# Resource Manager automatically:
# - Allocates to agents
# - Tracks consumption
# - Enforces limits (warn at 50%, throttle at 75%, stop at 95%)
# - Reallocates from idle agents
# - Forecasts exhaustion
```

---

## 🧩 Use Cases

This architecture is suitable for:

- **Multi-Agent Systems**: Coordinate multiple LLM agents with different roles
- **Workflow Automation**: Orchestrate complex, multi-step processes
- **Resource Optimization**: Track and optimize token/time/cost budgets
- **Production Monitoring**: Real-time observability of agent systems
- **Research**: Study meta-prompting and emergent agent behaviors

---

## 🔧 Extending the System

### Create a New Skill

```bash
# Use the skill generator meta-prompt
/generate "your capability description" --type=skill --level=N
```

The generator will create a skill following the standard format:
- Context (problem and solution)
- Capability (operations and examples)
- Constraints (rules and limits)
- Composition (how it integrates)
- Quality Metrics
- Anti-Patterns
- Examples

### Create a New Agent

```bash
# Use the agent generator meta-prompt
/generate "your agent description" --type=agent --level=L3_PLANNING
```

The generator will create an agent with:
- Mental Plane (understanding)
- Physical Plane (execution)
- Spiritual Plane (ethics)
- Interaction patterns
- Success criteria

### Create a New Command

```bash
# Use the command generator meta-prompt
/generate "your command description" --type=command
```

---

## 📊 Quality Metrics

All components include measurable quality metrics:

**Skills:**
- Specificity ≥ 0.90
- Composability ≥ 0.85
- Testability ≥ 0.80
- Documentability ≥ 0.85

**Agents:**
- Coverage ≥ 0.90 (% operations instrumented)
- Accuracy ≥ 0.95 (correctness)
- Latency P99 ≤ 100ms
- Throughput ≥ 100 ops/s

**System-Wide:**
- Resource utilization ≥ 85%
- Forecast accuracy ≥ 85%
- Monitoring overhead ≤ 5%

---

## 🌳 Repository Structure

```
agentic-skill-architecture/
├── README.md                   # This file
├── docs/                       # Comprehensive documentation
├── core/                       # Original meta-prompting theory
├── src/                        # Novel agentic architecture
│   ├── .claude/               # Skills, agents, commands
│   ├── meta-prompts/          # Generators
│   └── generators/            # Unified generator
├── iterations/                # 7-iteration workflows
└── examples/                  # Sample outputs
```

See [REPOSITORY_STRUCTURE_PROPOSAL.md](REPOSITORY_STRUCTURE_PROPOSAL.md) for detailed breakdown.

---

## 🤝 Contributing

Contributions welcome! See how to:

1. **Add a Skill**: Use the skill generator, follow quality standards
2. **Add an Agent**: Use the agent generator, ensure 3-plane architecture
3. **Add a Command**: Use the command generator, provide examples
4. **Improve Documentation**: PRs for docs/ always appreciated

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

This project builds on the [meta-prompting framework](https://github.com/original/repo), which is also MIT licensed.

---

## 🙏 Acknowledgments

- **Original Meta-Prompting Framework**: Foundation theory in `core/`
- **MERCURIO Agent Architecture**: Three-plane (Mental, Physical, Spiritual) agent design
- **7-Level Skill Hierarchy**: Progressive skill complexity with cognitive load modeling

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/agentic-skill-architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agentic-skill-architecture/discussions)

---

## 🔮 Future Directions

- [ ] Add more agentic skills (consensus, voting, auction)
- [ ] Create language-specific implementations (Python, TypeScript, Rust)
- [ ] Build visual workflow editor
- [ ] Add distributed deployment guide (Kubernetes, Docker)
- [ ] Create benchmarks and performance comparisons

---

**Built with ❤️ through meta-prompting**
