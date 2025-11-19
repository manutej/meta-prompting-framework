# Meta-Prompting Workflows

Pre-configured multi-agent workflows for common meta-prompting operations.

## Available Workflows

### 1. Meta-Framework Generation (`meta-framework-generation.yaml`)

**Purpose:** Generate a complete custom meta-prompting framework for any domain.

**Agents Used:**
- `deep-researcher` - Domain analysis
- `meta2` - Framework generation
- `MARS` - Validation across examples
- `mercurio-orchestrator` - Synthesis and documentation

**Steps:**
1. **Domain Analysis** - Research primitives, operations, complexity
2. **Categorical Mapping** - Map to category theory structures
3. **Framework Generation** - Create N-level framework
4. **Validation** - Test on concrete examples
5. **Synthesis** - Refine and document

**Parameters:**
- `domain_name` - Target domain (required)
- `depth_levels` - Number of levels (3-10, default: 5)
- `categorical_framework` - Structure type (default: natural_equivalence)
- `theoretical_depth` - Theory exposure (minimal/moderate/comprehensive)

**Example:**
```yaml
workflow: meta-framework-generation
parameters:
  domain_name: "API design patterns"
  depth_levels: 5
  categorical_framework: "inclusion"
  theoretical_depth: "moderate"
```

**Output:**
- Complete N-level framework
- Meta-prompts for each level
- Validation report
- Usage guide
- Examples

**Time:** 15-30 minutes
**Tokens:** 50K-100K

---

### 2. Quick Meta-Prompt Application (`quick-meta-prompt.yaml`)

**Purpose:** Fast application of production meta-prompts to any task.

**Agents Used:**
- `meta-agent` command

**Steps:**
1. **Complexity Analysis** - Assess task difficulty and domain
2. **Meta-Prompt Application** - Apply optimal V2 meta-prompt

**Parameters:**
- `task_description` - Task to enhance (required)
- `preferred_meta_prompt` - Override auto-selection (optional)
- `quality_priority` - speed/balanced/quality (default: balanced)

**Example:**
```yaml
workflow: quick-meta-prompt
parameters:
  task_description: "Design a rate limiter"
  quality_priority: "quality"
```

**Output:**
- Meta-prompt used
- Complexity analysis
- Reasoning trace
- Enhanced output

**Time:** 2-5 minutes
**Tokens:** 5K-15K

---

## Workflow Selection Guide

| Goal | Use Workflow | Why |
|------|--------------|-----|
| Need custom framework for domain | **meta-framework-generation** | Complete N-level framework |
| Quick task enhancement | **quick-meta-prompt** | Fast V2 meta-prompt application |

---

## Usage Patterns

### Pattern 1: Domain-Specific Framework

When you need a reusable framework for a specific domain:

```bash
# Generate framework once
run workflow: meta-framework-generation
  domain_name: "database optimization"
  depth_levels: 7

# Use generated framework repeatedly
framework.level(3).apply("optimize JOIN query")
framework.level(5).apply("design query execution engine")
```

### Pattern 2: One-Off Task Enhancement

When you have a single task to improve:

```bash
# Apply directly
run workflow: quick-meta-prompt
  task_description: "Implement authentication"
  quality_priority: "quality"
```

### Pattern 3: Progressive Enhancement

Start quick, escalate if needed:

```bash
# Try quick first
result1 = quick-meta-prompt("design API")

# If unsatisfied, generate full framework
if not result1.satisfactory():
    framework = meta-framework-generation("API design", 5)
    result2 = framework.level(4).apply("design API")
```

---

## Workflow Composition

Workflows can be chained:

```yaml
# Chain example
pipeline:
  - workflow: meta-framework-generation
    parameters:
      domain_name: "ML pipelines"
      depth_levels: 5
    save_output_as: ml_framework

  - workflow: quick-meta-prompt
    parameters:
      task_description: "Design feature engineering pipeline"
      use_framework: ${ml_framework}
```

---

## Creating Custom Workflows

Template for new workflows:

```yaml
---
name: Your Workflow Name
description: What it does

agents:
  - agent1
  - agent2

steps:
  - name: step1
    agent: agent1
    depends_on: []
    inputs: [...]
    outputs: [...]
    prompt: |
      Step prompt here

parameters:
  param1:
    type: string
    required: true
    description: What this parameter does

output:
  format: markdown
  save_to: "path/to/output.md"

tags:
  - tag1
  - tag2
---
```

**Best Practices:**
1. Clear step dependencies
2. Well-defined inputs/outputs
3. Descriptive prompts
4. Parameterization for reuse
5. Estimated time/tokens
6. Usage examples
7. Proper tags

---

## Workflow Parameters Reference

### Common Parameters

**domain_name** (string)
- Target domain for analysis
- Examples: "API design", "ML pipelines", "code refactoring"

**depth_levels** (integer, 3-10)
- Number of sophistication levels
- 3: Simple/Intermediate/Advanced
- 5: Novice/Competent/Proficient/Expert/Master
- 7: Full progression with Genius level

**categorical_framework** (enum)
- `natural_equivalence` - Elegant via Lemma 1 (default)
- `functors` - Explicit transformations
- `rewrite` - Task-agnosticity focus
- `inclusion` - Hierarchical embeddings
- `internal_hom` - Exponential objects
- `comprehensive` - All approaches

**theoretical_depth** (enum)
- `minimal` - Brief explanation
- `moderate` - Key proofs (default)
- `comprehensive` - Full treatment
- `research_level` - Novel contributions

**quality_priority** (enum)
- `speed` - Fast, good enough
- `balanced` - Speed/quality tradeoff (default)
- `quality` - Maximum quality, slower

---

## Output Formats

### Framework Output
```
generated-frameworks/
  ${domain_name}-framework.md
    ├── Executive Summary
    ├── Categorical Foundations
    ├── Level Architecture
    ├── Level Specifications (L1-LN)
    ├── Cross-Level Integration
    ├── Theoretical Justification
    ├── Usage Guide
    └── Examples
```

### Quick Meta-Prompt Output
```
outputs/
  ${task_id}-meta-enhanced.md
    ├── Meta-Prompt Used
    ├── Complexity Analysis
    ├── Reasoning Trace
    └── Final Output
```

---

## Troubleshooting

**Q: Workflow takes too long**
- Use `quick-meta-prompt` for faster results
- Reduce `depth_levels` (e.g., 3 instead of 7)
- Set `theoretical_depth: "minimal"`

**Q: Framework too complex**
- Reduce `depth_levels`
- Use simpler `categorical_framework` (try `inclusion`)
- Set `theoretical_depth: "minimal"`

**Q: Need more detail**
- Increase `depth_levels` (e.g., 7 or 10)
- Use `categorical_framework: "comprehensive"`
- Set `theoretical_depth: "comprehensive"`

**Q: Wrong meta-prompt selected**
- Override with `preferred_meta_prompt` parameter
- Adjust `quality_priority`
- Check complexity analysis output

---

## See Also

- [Agents](../agents/README.md) - Available agents
- [Commands](../commands/README.md) - Slash commands
- [Meta-Prompts V2](../meta-prompts/v2/META_PROMPTS.md) - Production meta-prompts
- [Quick Start](../docs/QUICK_START.md) - Get started guide

---

**Making meta-prompting workflows repeatable and composable.** ✨
