# Requirements Specification

**Document**: Formal requirements for MPG Core System
**Format**: FR (Functional), DR (Design), NFR (Non-Functional)

---

## Requirements Index

### Functional Requirements (FR)
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| FR-001 | Configuration Parsing | P0 | DEFINED |
| FR-002 | Schema Validation | P0 | DEFINED |
| FR-003 | Environment Variable Substitution | P0 | DEFINED |
| FR-004 | Parallel Site Generation | P0 | DEFINED |
| FR-005 | Template Scaffolding | P0 | DEFINED |
| FR-006 | Layout Expansion | P1 | DEFINED |
| FR-007 | CLI Command Interface | P0 | DEFINED |
| FR-008 | MCP Tool Exposure | P0 | DEFINED |
| FR-009 | Progress Tracking | P1 | DEFINED |
| FR-010 | Error Recovery | P1 | DEFINED |

### Design Requirements (DR)
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| DR-001 | Modular Architecture | P0 | DEFINED |
| DR-002 | Extensible Template System | P1 | DEFINED |
| DR-003 | Plugin-Ready Structure | P2 | DEFINED |
| DR-004 | Observable Execution | P1 | DEFINED |

### Non-Functional Requirements (NFR)
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| NFR-001 | Performance | P0 | DEFINED |
| NFR-002 | Reliability | P0 | DEFINED |
| NFR-003 | Security | P0 | DEFINED |
| NFR-004 | Maintainability | P1 | DEFINED |
| NFR-005 | Portability | P1 | DEFINED |

---

## Functional Requirements (FR)

### FR-001: Configuration Parsing

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL parse YAML configuration files containing site definitions, defaults, workflows, and integrations.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-001.1 | Valid YAML files load without errors | Unit test |
| FR-001.2 | Invalid YAML returns error with line number | Unit test |
| FR-001.3 | UTF-8 encoding supported | Unit test |
| FR-001.4 | File paths resolved relative to config location | Integration test |
| FR-001.5 | Config file watch mode available (optional) | Feature flag |

#### Dependencies
- None (foundational)

#### Test Scenarios
```gherkin
Scenario: Parse valid configuration
  Given a valid sites.yaml file
  When the config loader reads the file
  Then it returns a typed MPGConfig object
  And no errors are thrown

Scenario: Handle invalid YAML syntax
  Given a sites.yaml with invalid YAML syntax
  When the config loader reads the file
  Then it throws a ConfigParseError
  And the error includes the line number
  And the error includes a helpful message
```

---

### FR-002: Schema Validation

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL validate parsed configuration against the Zod schema, rejecting invalid configurations with specific field-level errors.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-002.1 | Missing required fields return specific error | Unit test |
| FR-002.2 | Invalid field types return type mismatch error | Unit test |
| FR-002.3 | Invalid enum values list allowed values | Unit test |
| FR-002.4 | Nested validation errors include full path | Unit test |
| FR-002.5 | Unknown fields generate warnings (not errors) | Unit test |

#### Dependencies
- FR-001 (Configuration Parsing)

#### Test Scenarios
```gherkin
Scenario: Validate required fields
  Given a config missing the 'id' field for a site
  When validation runs
  Then it returns ValidationError
  And error.path equals "sites[0].id"
  And error.message contains "required"

Scenario: Allow unknown fields with warning
  Given a config with extra field "custom_field"
  When validation runs
  Then config is accepted
  And a warning is logged about unknown field
```

---

### FR-003: Environment Variable Substitution

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL substitute `${VAR_NAME}` patterns with environment variable values before validation.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-003.1 | `${VAR}` replaced with process.env.VAR | Unit test |
| FR-003.2 | Missing required env var throws EnvVarError | Unit test |
| FR-003.3 | `${VAR:-default}` syntax for defaults | Unit test |
| FR-003.4 | Nested substitution not supported (security) | Unit test |
| FR-003.5 | Substitution happens before validation | Integration test |

#### Dependencies
- FR-001 (Configuration Parsing)

#### Test Scenarios
```gherkin
Scenario: Substitute environment variable
  Given BUILDER_API_KEY="sk-test-123" in environment
  And config contains api_key: "${BUILDER_API_KEY}"
  When config is loaded
  Then api_key equals "sk-test-123"

Scenario: Handle missing required env var
  Given MISSING_VAR is not in environment
  And config contains required: "${MISSING_VAR}"
  When config is loaded
  Then EnvVarError is thrown
  And error.variable equals "MISSING_VAR"
```

---

### FR-004: Parallel Site Generation

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL generate multiple sites concurrently with configurable parallelism limits.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-004.1 | Concurrency configurable from 1-50 | Config validation |
| FR-004.2 | Default concurrency is 10 | Unit test |
| FR-004.3 | Failed sites do not block other sites | Integration test |
| FR-004.4 | Each site generates independently | Integration test |
| FR-004.5 | Total execution time scales with concurrency | Benchmark |

#### Dependencies
- FR-001, FR-002, FR-005

#### Test Scenarios
```gherkin
Scenario: Parallel generation with failures
  Given 5 sites configured
  And site 3 has invalid template reference
  When /mpg apply --all runs with concurrency=5
  Then sites 1,2,4,5 generate successfully
  And site 3 reports failure
  And execution completes (not blocked)

Scenario: Concurrency limits execution
  Given 10 sites configured
  And concurrency set to 2
  When /mpg apply --all runs
  Then at most 2 sites generate simultaneously
  And all 10 sites eventually complete
```

---

### FR-005: Template Scaffolding

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL scaffold sites by copying template directories and substituting variables.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-005.1 | Template directory copied to output location | Integration test |
| FR-005.2 | `{{variable}}` patterns replaced in all files | Unit test |
| FR-005.3 | `__name__` file patterns renamed | Unit test |
| FR-005.4 | Binary files copied without modification | Unit test |
| FR-005.5 | Template metadata (template.yaml) not copied | Unit test |

#### Dependencies
- FR-001, FR-002

#### Test Scenarios
```gherkin
Scenario: Scaffold site from template
  Given template "next-marketing" exists
  And site config has name="Alpha" and palette="emerald"
  When scaffold step runs
  Then output directory contains all template files
  And {{site.name}} replaced with "Alpha"
  And {{brand.palette}} replaced with "emerald"
  And package.json name is "alpha"
```

---

### FR-006: Layout Expansion

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Description
The system SHALL expand compact layout notation (`hero+features+cta`) into component import statements and page structure.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-006.1 | `hero+features+cta` parses to 3 components | Unit test |
| FR-006.2 | Unknown component names generate warning | Unit test |
| FR-006.3 | Component order preserved | Unit test |
| FR-006.4 | Generates valid JSX/TSX structure | Integration test |

#### Dependencies
- FR-005 (Template Scaffolding)

#### Test Scenarios
```gherkin
Scenario: Expand layout notation
  Given layout: "hero+features+cta"
  When layout expansion runs
  Then output includes: import Hero from './components/hero'
  And output includes: import Features from './components/features'
  And output includes: import CTA from './components/cta'
  And JSX order is: <Hero /><Features /><CTA />
```

---

### FR-007: CLI Command Interface

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL provide CLI commands following the `/mpg verb target modifiers` grammar.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-007.1 | `mpg list` shows all sites | Integration test |
| FR-007.2 | `mpg plan` shows dry-run output | Integration test |
| FR-007.3 | `mpg apply` executes generation | Integration test |
| FR-007.4 | `mpg status` shows job progress | Integration test |
| FR-007.5 | `mpg view` shows site details | Integration test |
| FR-007.6 | `--help` generates usage for all commands | Unit test |
| FR-007.7 | Exit codes: 0=success, 1=error | Integration test |

#### Dependencies
- FR-001 through FR-006

#### Test Scenarios
```gherkin
Scenario: List sites with filter
  Given sites.yaml with 5 marketing and 3 docs sites
  When mpg list sites type=marketing runs
  Then output shows 5 sites
  And output format is table by default

Scenario: Plan shows dry-run
  Given valid sites.yaml
  When mpg plan --dry runs
  Then output shows what would be generated
  And no files are created
  And exit code is 0
```

---

### FR-008: MCP Tool Exposure

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL expose all CLI commands as MCP tools callable by AI agents.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-008.1 | `list_sites` tool returns site array | MCP test |
| FR-008.2 | `plan_sites` tool returns plan object | MCP test |
| FR-008.3 | `apply_sites` tool triggers generation | MCP test |
| FR-008.4 | Tool responses are valid JSON | Unit test |
| FR-008.5 | Errors include actionable messages | Unit test |

#### Dependencies
- FR-007 (CLI Interface)

#### Test Scenarios
```gherkin
Scenario: MCP list_sites tool
  Given MCP server is running
  And valid sites.yaml is loaded
  When client calls list_sites tool
  Then response is JSON array
  And each item has id, name, type fields
```

---

### FR-009: Progress Tracking

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Description
The system SHALL track and report progress during multi-site generation.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-009.1 | Per-site status available (pending/running/done/failed) | Integration test |
| FR-009.2 | Overall progress percentage calculated | Unit test |
| FR-009.3 | Estimated time remaining calculated | Unit test |
| FR-009.4 | Progress updates emitted as events | Unit test |

---

### FR-010: Error Recovery

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Description
The system SHALL recover gracefully from errors, continuing with remaining sites.

#### Acceptance Criteria
| ID | Criterion | Verification |
|----|-----------|--------------|
| FR-010.1 | Failed sites logged with error details | Integration test |
| FR-010.2 | Successful sites not affected by failures | Integration test |
| FR-010.3 | Final summary shows success/failure counts | Integration test |
| FR-010.4 | Partial success is valid outcome (exit code 0) | Integration test |

---

## Design Requirements (DR)

### DR-001: Modular Architecture

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Description
The system SHALL be organized into independent, testable modules with clear boundaries.

#### Requirements
| ID | Requirement |
|----|-------------|
| DR-001.1 | Core logic independent of CLI framework |
| DR-001.2 | MCP server wraps core, doesn't duplicate |
| DR-001.3 | Each module has single responsibility |
| DR-001.4 | Dependencies flow inward (core has no external deps) |

#### Module Structure
```
src/
├── core/           # Business logic (no framework deps)
│   ├── config/     # FR-001, FR-002, FR-003
│   ├── generator/  # FR-005, FR-006
│   └── orchestrator/ # FR-004, FR-009, FR-010
├── cli/            # FR-007 (Commander.js)
└── mcp/            # FR-008 (MCP SDK)
```

---

### DR-002: Extensible Template System

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Description
The system SHALL support custom templates without code changes.

#### Requirements
| ID | Requirement |
|----|-------------|
| DR-002.1 | Templates are directories, not code |
| DR-002.2 | Template metadata in `template.yaml` |
| DR-002.3 | Custom templates in `templates/` directory |
| DR-002.4 | Templates discoverable at runtime |

---

### DR-003: Plugin-Ready Structure

**Priority**: P2 (Nice to Have)
**Status**: DEFINED

#### Description
The system SHALL be structured to support plugins in future versions.

#### Requirements
| ID | Requirement |
|----|-------------|
| DR-003.1 | Clear extension points documented |
| DR-003.2 | Event system for lifecycle hooks |
| DR-003.3 | Plugin interface defined (not implemented in v1) |

---

### DR-004: Observable Execution

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Description
The system SHALL emit events/logs enabling debugging and monitoring.

#### Requirements
| ID | Requirement |
|----|-------------|
| DR-004.1 | Structured logging (JSON format available) |
| DR-004.2 | Log levels: debug, info, warn, error |
| DR-004.3 | Execution trace for debugging |
| DR-004.4 | Timing information for performance analysis |

---

## Non-Functional Requirements (NFR)

### NFR-001: Performance

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start | <2s | Time to first command output |
| 10-site scaffold | <60s | End-to-end with concurrency=10 |
| Memory (10 sites) | <500MB | Peak RSS |
| Memory (100 sites) | <2GB | Peak RSS |

---

### NFR-002: Reliability

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Success rate | >99% | Valid config → successful generation |
| Error handling | 100% | All errors caught and reported |
| Crash recovery | Graceful | No corrupted output on crash |

---

### NFR-003: Security

**Priority**: P0 (Must Have)
**Status**: DEFINED

#### Requirements
| ID | Requirement |
|----|-------------|
| NFR-003.1 | Secrets never logged (pattern redaction) |
| NFR-003.2 | No arbitrary code execution (no eval) |
| NFR-003.3 | Path traversal prevented |
| NFR-003.4 | Dependencies audited (npm audit) |

---

### NFR-004: Maintainability

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Metrics
| Metric | Target |
|--------|--------|
| Test coverage | >80% |
| Documentation | All public APIs |
| Code complexity | Cyclomatic <10 per function |

---

### NFR-005: Portability

**Priority**: P1 (Should Have)
**Status**: DEFINED

#### Requirements
| ID | Requirement |
|----|-------------|
| NFR-005.1 | macOS (ARM + Intel) supported |
| NFR-005.2 | Linux x64 supported |
| NFR-005.3 | Windows via WSL supported |
| NFR-005.4 | Node.js 20+ required |

---

## Traceability Matrix

| Requirement | User Story | Test | ADR |
|-------------|------------|------|-----|
| FR-001 | P1-Generate | T0.4 | - |
| FR-002 | P1-Generate | T0.4 | ADR-002 |
| FR-003 | P1-Generate | T0.5 | - |
| FR-004 | P1-Generate | T0.10 | ADR-003 |
| FR-005 | P1-Generate | T0.8 | ADR-005 |
| FR-006 | P1-Generate | T0.9 | - |
| FR-007 | P1-Preview, P2-Status | T0.13-18 | ADR-004 |
| FR-008 | P1-MCP | T0.21-24 | ADR-006 |
| FR-009 | P2-Status | T0.11 | - |
| FR-010 | P1-Generate | T0.12 | - |

---

*Requirements are versioned. Changes require spec amendment and approval.*
