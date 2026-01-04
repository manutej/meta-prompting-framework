# Feature Specification Template

## Instructions

This template guides writing clear, unambiguous specifications using MERCURIO three-plane analysis. Follow these rules:

1. **Focus on WHAT and WHY, not HOW** - Implementation details go in plan.md
2. **Mark ambiguities** with `[NEEDS CLARIFICATION]`
3. **Document assumptions** with `[ASSUMPTION]` markers
4. **Use Gherkin syntax** for acceptance tests (Given/When/Then)
5. **Apply three-plane analysis** to surface risks and conflicts

---

# Feature Specification: [Feature Name]

**Version:** 0.1.0-draft
**Status:** DRAFTING | REVIEW | APPROVED
**Last Updated:** YYYY-MM-DD

---

## Executive Summary

**One Sentence**: [What does this feature enable? Who benefits?]

**MERCURIO Wisdom Check**:
- **Mental**: [Is this intellectually sound?]
- **Physical**: [Is this feasible?]
- **Spiritual**: [Does this serve users?]

---

## 1. MENTAL PLANE: Understanding

### 1.1 Problem Analysis

**Validated Problem**: [What specific problem are we solving?]

**Evidence**:
- [VALIDATED] Evidence that supports this problem exists
- [ASSUMPTION] Assumption that needs validation
- [NEEDS RESEARCH] Area requiring investigation

### 1.2 Assumptions Registry

| ID | Assumption | Validation Method | Status |
|----|------------|-------------------|--------|
| A1 | [Assumption] | [How to validate] | UNVALIDATED/PARTIAL/VALIDATED |

### 1.3 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| [Metric] | [Specific target] | [How to measure] |

---

## 2. PHYSICAL PLANE: Constraints

### 2.1 Resource Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| [Constraint] | [How it affects scope] | [How to handle] |

### 2.2 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| [Risk] | Low/Medium/High | Low/Medium/High | [Mitigation] |

### 2.3 MVP Scope

**IN SCOPE** (must ship):
- [Essential feature]

**OUT OF SCOPE** (Phase 1+):
- [Deferred feature] - Reason: [Why deferred]

---

## 3. SPIRITUAL PLANE: Purpose

### 3.1 User Personas

**PRIMARY (v1 Target)**:
```yaml
name: [Name]
role: [Role]
problem: "[Their specific pain point]"
desired_outcome: "[What success looks like]"
```

**SECONDARY (v2 Target)**:
```yaml
name: [Name]
role: [Role]
problem: "[Their specific pain point]"
```

### 3.2 Values Alignment

| Value | How Feature Honors It |
|-------|----------------------|
| [Value] | [How] |

### 3.3 Unintended Consequences

| Consequence | Type | Mitigation |
|-------------|------|------------|
| [Potential consequence] | Positive/Negative/Neutral | [How to handle] |

---

## 4. THREE-PLANE CONVERGENCE

### 4.1 Where All Planes Agree (Build First)

| Feature | Mental | Physical | Spiritual |
|---------|--------|----------|-----------|
| [Feature] | [Why true] | [Why feasible] | [Why right] |

### 4.2 Where Planes Conflict (Risk Zones)

#### Conflict: [Name]

| Plane | Position |
|-------|----------|
| Mental | "[Mental plane view]" |
| Physical | "[Physical plane view]" |
| Spiritual | "[Spiritual plane view]" |

**Resolution**: [How to resolve this conflict]

### 4.3 Wisdom Synthesis

> "[One-paragraph integrated recommendation based on all three planes]"

---

## 5. User Stories

### P1 (Must Have) - [Story Title]

**User**: [Persona name]

**Journey**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Success Criteria**: [Specific, measurable outcome]

**Acceptance Tests**:
```gherkin
Given [initial state]
When [action]
Then [observable outcome]
And [additional outcome]
```

**Unknowns**:
- [NEEDS CLARIFICATION] [What's unclear?]

---

### P2 (Should Have) - [Story Title]

[Same structure as P1]

---

### P3 (Nice to Have) - [Story Title]

[Same structure as P1]

---

## 6. Functional Requirements

### FR-001: [Requirement Name]

The system MUST [specific capability].

**Acceptance Criteria**:
- [Criterion 1]
- [Criterion 2]

**Affected Stories**: P1 ([Story]), P2 ([Story])

---

## 7. Key Entities

```typescript
// Brief type definitions
interface EntityName {
  field: type;
}
```

*Full types in data-model.md*

---

## 8. Out of Scope (Explicit)

| Feature | Reason | Phase |
|---------|--------|-------|
| [Feature] | [Why out of scope] | Phase N |

---

## 9. Open Questions

| ID | Question | Status | Blocking? |
|----|----------|--------|-----------|
| Q1 | [Question] | NEEDS INPUT/DECIDED/DEFER | Yes/No |

---

## 10. Approval Checklist

### Specification Phase Gate

- [ ] All P1 stories have acceptance tests
- [ ] No unresolved `[NEEDS CLARIFICATION]` markers
- [ ] Assumptions documented with validation methods
- [ ] Three-plane analysis complete
- [ ] Out-of-scope explicitly listed
- [ ] Open questions resolved or marked DEFER

### Approvers

| Role | Status | Date |
|------|--------|------|
| [Role] | PENDING/APPROVED | |

---

*Proceed to plan.md only after specification approval.*
