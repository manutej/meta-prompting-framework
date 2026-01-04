# Research: CLI Framework Libraries

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-004

---

## Executive Summary

**Recommendation**: Commander.js for MPG CLI (with oclif as scale alternative)

Commander.js provides the best balance of simplicity, ecosystem, and features for a 10-command CLI. oclif becomes valuable at 20+ commands or when plugins are needed.

---

## Comparison Matrix

| Framework | GitHub Stars | TypeScript | Subcommands | Plugins | Best For |
|-----------|-------------|------------|-------------|---------|----------|
| **Commander** | 26.2k | Good | Excellent | No | MPG ✓ |
| **oclif** | 8.9k | Native | Excellent | Yes | Enterprise |
| **yargs** | 11k | Good | Good | No | Complex args |
| **citty** | 878 | Good | Yes | Composable | UnJS |
| **CAC** | Varied | Native | Yes | No | Minimal |
| **Clipanion** | 2.5k | Native | Excellent | Yes | Type-safe |
| **meow** | Popular | Good | No | No | Simple |
| **arg** | 1.5k | Good | No | No | Bare-bones |

---

## Feature Comparison

| Feature | Commander | oclif | yargs | CAC | Clipanion |
|---------|-----------|-------|-------|-----|-----------|
| Help generation | Excellent | Excellent | Good | Auto | Auto |
| Learning curve | Easy | Medium | Medium | Easy | Medium |
| Plugin system | No | Yes | No | No | Yes |
| Interactive prompts | External | External | Extension | External | External |
| Dependencies | Minimal | Minimal | Moderate | Zero | Zero |
| Real-world usage | Excellent | Enterprise | Excellent | Good | Yarn |

---

## Detailed Analysis

### Commander.js (Recommended)

**Pros:**
- Most popular (26.2k stars)
- Excellent help generation
- Clean, intuitive API
- Great documentation
- Easy to learn (hours, not days)
- Works for 5-15 commands perfectly

**Cons:**
- No plugin system
- Interactive prompts require Inquirer.js
- Manual output formatting (use Chalk)

**Example:**
```typescript
import { Command } from 'commander';

const program = new Command();

program
  .name('mpg')
  .description('Multi-Project Generator')
  .version('0.1.0');

program
  .command('list')
  .description('List configured sites')
  .option('-t, --type <type>', 'Filter by site type')
  .option('--format <format>', 'Output format', 'table')
  .action((options) => listSites(options));

program
  .command('apply')
  .description('Generate sites')
  .option('--ids <ids>', 'Comma-separated site IDs')
  .option('--steps <steps>', 'Workflow steps (e.g., scaffold+design)')
  .action((options) => applySites(options));

program.parse();
```

### oclif (Scale Alternative)

**Pros:**
- Enterprise-grade (Heroku, Salesforce, Twilio)
- Excellent plugin architecture
- Built-in testing utilities
- Autocomplete generation
- Documentation generation

**Cons:**
- 2-3 day setup vs hours for Commander
- More opinionated structure
- Heavier for small CLIs

**When to choose:** 20+ commands, plugin system needed, enterprise deployment

### CAC (Minimal Alternative)

**Pros:**
- Zero dependencies
- Written in TypeScript
- Dot-nested options (`--env.API_KEY`)
- Simple 4-method API

**Cons:**
- Smaller ecosystem
- Less documentation

**When to choose:** Bundle size critical, minimal dependency preference

---

## Interactive Prompts

All frameworks pair with external prompt libraries:

| Library | Style | Best For |
|---------|-------|----------|
| **Inquirer.js** | Full-featured | Complex prompts |
| **prompts** | Lightweight | Simple prompts |
| **enquirer** | Elegant | Modern UX |

---

## Output Formatting

Pair with:
- **Chalk** - Colors and styles
- **Table** - Structured data
- **ora** - Spinners and loading
- **ink** - React-like CLI components

---

## Recommendation for MPG

### Phase 0 (MVP): Commander.js

**Rationale:**
1. Best balance of simplicity and features
2. Excellent documentation and community
3. 10 commands is perfect fit
4. Easy to add prompts (Inquirer) and colors (Chalk)
5. Can migrate to oclif later if needed

### Commands to Implement:
```
mpg list [sites] [--type] [--format]
mpg plan [sites] [--type] [--steps] [--dry]
mpg apply [sites] [--ids] [--steps] [--concurrency]
mpg status [--job]
mpg view [site] [--plane]
mpg run <workflow> [--type]
mpg init
mpg --help
mpg --version
```

### Phase 2+ (Scale): Consider oclif

**When:**
- 20+ commands
- Plugin system needed
- Multiple CLI distributions

---

## Decision

**CONFIRMED: Commander.js**

Rationale:
1. Perfect fit for 10 commands
2. Excellent help generation
3. Minimal learning curve
4. Proven at scale (thousands of CLIs)
5. Easy integration with Inquirer.js for prompts

---

## Sources

- [Commander.js Documentation](https://tj.github.io/commander.js/)
- [oclif: The Open CLI Framework](https://oclif.io/)
- [Building a CLI with Node.js in 2024](https://egmz.medium.com/building-a-cli-with-node-js-in-2024)
- [Heroku CLI v9 oclif Migration](https://www.heroku.com/blog/heroku-cli-v9)
