# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-29

### Added

Initial release as Claude Code plugin following official plugin architecture.

#### Plugin Structure
- `.claude-plugin/plugin.json` manifest with full metadata
- Proper plugin directory layout per official spec

#### Skills (6 total, namespaced `/ilf:*`)
- `/ilf:witness` - Construct existence witnesses using intuitionistic logic
- `/ilf:first-principles` - Musk-style deconstruction to verified axioms
- `/ilf:contrarian` - Thiel-style search for non-obvious truths
- `/ilf:analyze-logic` - Classical vs intuitionistic validity analysis
- `/ilf:prove` - Generate constructive proofs via Curry-Howard
- `/ilf:check-type` - Type-level verification

#### Agents (3 total)
- `witness-constructor` - Specialist for building witnesses
- `logic-analyzer` - Evaluates arguments under multiple logics
- `proof-verifier` - Validates constructive proofs

#### Documentation
- Comprehensive README with examples
- Detailed SKILL.md for each command
- Agent specifications with operating principles
- CHANGELOG for version tracking

### Philosophy
Applies three paradigms to constructive logic:
- **Musk Mode**: First-principles engineering reasoning
- **Thiel Mode**: Contrarian epistemology
- **Ramanujan Mode**: Divine intuition + rigorous verification

### Known Limitations
- Not yet battle-tested in real Claude Code sessions
- Namespace behavior (`/ilf:*`) needs verification
- Edge cases and error handling may need refinement

---

## Version History

### [0.1.0] - 2024-11-29
Initial Claude Code plugin release.

**Next Planned:**
- v0.2.0: Add examples directory with TypeScript/Haskell code
- v0.3.0: Add theory documentation (BHK, Curry-Howard deep dives)
- v0.4.0: Integration tests and CI/CD
- v1.0.0: Battle-tested, published to plugin marketplace

---

## The Constructive Principle for Versioning

Each version must provide a **witness** of improvement:
- Not "better" but "here's the specific capability added"
- Not "improved" but "here's the bug fixed with reproduction steps"
- Not "enhanced" but "here's the benchmark"

The changelog IS the proof of progress.
