# Multi-Project Site Generator - Constitution

**Version:** 1.0.0
**Status:** FOUNDATIONAL DOCUMENT
**Governance:** Amendments require explicit approval

---

## Preamble

This constitution establishes the core principles, non-negotiable articles, and governance model for the Multi-Project Site Generator (MPG). All specifications, plans, and implementations MUST align with these principles.

**MERCURIO Wisdom**: Before building, ask three questions:
- **Mental**: Is this true? (Intellectually sound, evidence-based)
- **Physical**: Can we do this? (Feasible, realistic, resourced)
- **Spiritual**: Is this right? (Serves users, aligns with values)

---

## Core Principles

### Principle 1: Intent Over Implementation

**Focus on WHAT and WHY before HOW.**

Specifications describe user needs and success criteria. Implementation details belong in plans, not specs. This prevents premature optimization and keeps focus on user value.

*Violation example*: Specifying "use Redis for caching" instead of "system should handle 100 concurrent generations"

### Principle 2: Terminal-First, Visual-Optional

**CLI is the primary interface. Visual builders are enhancements, not dependencies.**

The tool must be fully functional from terminal alone. Visual builder integration (Builder.io, Plasmic) is valuable but optional. Users should never be forced into a GUI workflow.

*Violation example*: Requiring Builder.io account to use basic generation features

### Principle 3: Parallel by Default, Sequential by Choice

**Design for concurrent execution. Single-site mode is a special case of multi-site.**

The orchestrator assumes parallelism. Concurrency is configurable (1-50). Sequential execution is achieved by setting concurrency=1, not through separate code paths.

*Violation example*: Separate `generate_single()` and `generate_batch()` implementations

### Principle 4: MCP-Native Features

**Every capability exposed via CLI is also available as an MCP tool.**

Claude Code and other AI agents must have full access to MPG functionality. This enables agentic workflows where AI can orchestrate site generation autonomously.

*Violation example*: CLI-only commands without MCP equivalents

### Principle 5: Right-Sized Solutions

**Build the minimum solution that solves the core problem. Elegance is not a goal.**

Prefer simple over sophisticated. Features must justify their complexity with clear user benefit. Mathematical elegance does not equal user value.

*Violation example*: Implementing 4 abstraction planes when 1 would suffice for v1

---

## Development Articles

### Article I: Independent Testability

Every user story MUST be independently testable. If a feature cannot be verified in isolation, it is poorly scoped.

**Test**: Can you write a Gherkin scenario (Given/When/Then) for this feature?

### Article II: Explicit Assumptions

All assumptions MUST be documented with `[ASSUMPTION]` markers. Unvalidated assumptions MUST include validation methods.

**Test**: Can you point to evidence for each claim in the specification?

### Article III: Honest Timelines

Time estimates MUST be realistic, not optimistic. Use 1.5x multiplier on initial estimates. If a feature cannot be implemented in the timeline, it is out of scope.

**Test**: Would a single experienced developer complete this in the estimated time?

### Article IV: MVP First

Phase 0 (MVP) MUST be achievable in 8-10 weeks by one developer. Features not essential for MVP go to Phase 1+.

**Test**: Is this feature required for the PRIMARY persona's core workflow?

### Article V: No Vendor Lock-In

Core functionality MUST NOT require specific external services. Visual builders, CMS providers, and hosting platforms are optional integrations.

**Test**: Can a user run MPG without signing up for any third-party service?

### Article VI: Security by Design

API keys and secrets MUST NEVER appear in logs, configs (use `${ENV_VAR}` substitution), or error messages. Validate all input at system boundaries.

**Test**: Could running this command expose secrets?

### Article VII: Complexity Budget

Each phase has a complexity budget:
- **Phase 0 (MVP)**: 3 core components maximum
- **Phase 1**: Add 2 components maximum
- **Phase 2+**: Evaluate based on user feedback

**Test**: Can you explain the architecture in a 5-minute conversation?

---

## Amendment Protocol

Principles and Articles may be amended with:

1. **Proposal**: Written justification with three-plane analysis (Mental/Physical/Spiritual)
2. **Evidence**: User feedback, metrics, or research supporting the change
3. **Impact**: Assessment of how amendment affects existing specifications
4. **Approval**: Explicit approval from project maintainers

Amendments MUST NOT violate the Preamble's core questions (true/doable/right).

---

## Constitution Checklist

Before proceeding from Spec → Plan → Tasks → Implementation, verify:

### Specification Checkpoint
- [ ] All user stories are P1/P2/P3 prioritized
- [ ] No `[NEEDS CLARIFICATION]` markers remain
- [ ] Each requirement has acceptance criteria
- [ ] Assumptions are documented with validation methods
- [ ] Out-of-scope items are explicitly listed

### Plan Checkpoint
- [ ] Architecture respects all Articles
- [ ] ADRs (Architecture Decision Records) document major choices
- [ ] Timeline is realistic (Article III)
- [ ] Complexity budget respected (Article VII)
- [ ] No vendor lock-in (Article V)

### Tasks Checkpoint
- [ ] Every requirement maps to a task
- [ ] Dependencies are explicit
- [ ] Parallelizable tasks marked with `[P]`
- [ ] Test scenarios included for each feature
- [ ] No single task exceeds 2 days of work

---

## Appendix: MERCURIO Three-Plane Validation

Use this framework for major decisions:

```
┌─────────────────────────────────────────────────────────┐
│                     DECISION: [X]                        │
├─────────────────────────────────────────────────────────┤
│ MENTAL PLANE: Is this true?                             │
│ - Evidence supporting this decision:                    │
│ - Assumptions made:                                     │
│ - What could be wrong:                                  │
├─────────────────────────────────────────────────────────┤
│ PHYSICAL PLANE: Can we do this?                         │
│ - Resources required:                                   │
│ - Timeline estimate:                                    │
│ - Risks and mitigations:                                │
├─────────────────────────────────────────────────────────┤
│ SPIRITUAL PLANE: Is this right?                         │
│ - Who benefits:                                         │
│ - Unintended consequences:                              │
│ - Values alignment:                                     │
├─────────────────────────────────────────────────────────┤
│ CONVERGENCE: Do all three planes agree?                 │
│ - [ ] Mental: Approved                                  │
│ - [ ] Physical: Approved                                │
│ - [ ] Spiritual: Approved                               │
│ - Wisdom synthesis:                                     │
└─────────────────────────────────────────────────────────┘
```

---

*This Constitution governs all MPG development. When in doubt, return to these principles.*
