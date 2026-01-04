# Research: Template & Scaffolding Systems

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-005

---

## Executive Summary

**Recommendation**: Hybrid approach - giget + Handlebars + custom orchestrator

For batch site generation (10-100 sites), no single tool is ideal. The optimal approach combines giget for template download, Handlebars for variable substitution, and a custom orchestrator for parallel execution.

---

## Comparison Matrix

| Tool | Type | Batch-Ready | Binary Safe | Complexity |
|------|------|-------------|-------------|------------|
| **Custom copy+replace** | DIY | Yes | Yes | Simple |
| **Plop** | Interactive | No | Limited | Simple |
| **Hygen** | Scalable | Partial | Limited | Medium |
| **Yeoman** | Full framework | No | Yes | Complex |
| **EJS** | Template engine | Component | No | Simple |
| **Handlebars** | Template engine | Component | No | Simple |
| **giget** | Downloader | Yes | Yes | Simple |
| **degit** | Downloader | Partial | Yes | Simple |

---

## Detailed Analysis

### Simple File Copy + String Replace (Recommended Component)

**Pros:**
- Complete control
- Zero dependencies
- Perfect for straightforward needs
- 50-200 lines of code

**Cons:**
- You maintain the code
- No built-in conditional logic
- Limited to simple patterns

**Implementation:**
```typescript
async function scaffoldSite(template: string, site: SiteConfig, outDir: string) {
  // 1. Copy template directory
  await copy(templateDir, outDir);

  // 2. Walk text files, substitute {{variables}}
  for (const file of await glob(`${outDir}/**/*.{ts,tsx,json,md}`)) {
    let content = await readFile(file, 'utf-8');
    content = content.replace(/\{\{(\w+(?:\.\w+)*)\}\}/g, (_, path) =>
      get(site, path) ?? ''
    );
    await writeFile(file, content);
  }

  // 3. Rename __pattern__ files
  for (const file of await glob(`${outDir}/**/*__*__*`)) {
    const newName = file.replace(/__(\w+)__/g, (_, key) => site[key]);
    await rename(file, newName);
  }
}
```

### giget (Recommended for Template Download)

**Pros:**
- Modern, well-maintained (UnJS)
- Fast tarball downloads
- CLI + programmatic API
- Multi-provider (GitHub, GitLab, Bitbucket)
- 4.2M weekly downloads

**Cons:**
- Just downloads, no templating
- Requires post-processing

**Usage:**
```typescript
import { downloadTemplate } from 'giget';

await downloadTemplate('github:my-org/site-template', {
  dir: outputDir,
  install: false
});
```

### Handlebars (Recommended Template Engine)

**Pros:**
- Logic-less (safer than EJS)
- Familiar `{{variable}}` syntax
- Custom helpers for extensibility
- Used by Plop internally

**Cons:**
- Not a scaffolding framework
- Needs custom orchestration

**Usage:**
```typescript
import Handlebars from 'handlebars';

const template = Handlebars.compile(fileContent);
const result = template({
  siteName: site.name,
  brandPalette: site.brand.palette
});
```

### Plop (Interactive Generator)

**Pros:**
- Interactive prompts (Inquirer-based)
- Handlebars built-in
- Great for team consistency

**Cons:**
- **Not batch-ready** - requires user input
- Can't generate 100 sites in parallel
- Focuses on single components

**When to use:** Developer tooling (generate one component), not MPG

### Hygen (Scalable Generator)

**Pros:**
- EJS templating (powerful)
- Scalable for teams
- Can inject code into files
- Templates live with code

**Cons:**
- EJS security concerns
- Not designed for batch site generation
- Missing orchestration

**When to use:** Component generation within a project

### Yeoman (Full Framework)

**Pros:**
- Mature (since 2012)
- 5600+ community generators
- Full project scaffolding

**Cons:**
- Heavy and dated
- Interactive only
- Slow cold starts

**When to use:** Enterprise with existing Yeoman investment

---

## Framework Scaffolding Patterns

### How Next.js scaffolds:
```
create-next-app → Interactive CLI → Single project
```

### How Remix scaffolds:
```
create-remix → Interactive CLI → Single project
```

### How Astro scaffolds:
```
create-astro → Interactive CLI → Single project
```

**Key insight:** All modern frameworks use interactive single-project CLIs. None are designed for batch generation.

---

## Variable Substitution Patterns

| Pattern | Tool(s) | Example |
|---------|---------|---------|
| `{{variable}}` | Handlebars, Mustache | `{{siteName}}` |
| `<%= variable %>` | EJS | `<%= site.name %>` |
| `${VARIABLE}` | Env vars | `${API_KEY}` |
| `__variable__` | File names | `__siteName__.tsx` |

**Recommendation for MPG:**
- `{{variable}}` for template content (Handlebars)
- `${VAR}` for environment variables (pre-substitution)
- `__name__` for file/directory renames

---

## Binary File Handling

| Tool | Binary Safe | Method |
|------|-------------|--------|
| giget/degit | Yes | Downloads unchanged |
| Custom | Yes | File type detection |
| EJS/Handlebars | No | Text-only engines |
| Plop | Limited | Workarounds needed |

**Rule:** Never run template engines on images, fonts, or PDFs.

---

## Recommendation for MPG

### Architecture:
```
MPG Template Pipeline:
├── Phase 1: Download (giget)
├── Phase 2: Variable substitution (Handlebars)
├── Phase 3: File renames (custom)
├── Phase 4: Binary copy (custom)
└── Phase 5: Parallel orchestration (p-queue)
```

### Why Hybrid:
1. **giget** - Fast, parallel-friendly downloads
2. **Handlebars** - Safe, familiar syntax
3. **Custom orchestrator** - Parallel execution for 100 sites
4. **Minimal dependencies** - No heavy frameworks

### Code Example:
```typescript
import { downloadTemplate } from 'giget';
import Handlebars from 'handlebars';

async function generateSites(sites: SiteConfig[], concurrency = 10) {
  const queue = new PQueue({ concurrency });

  for (const site of sites) {
    queue.add(async () => {
      // 1. Download template
      await downloadTemplate(`github:org/template`, { dir: site.outDir });

      // 2. Process text files with Handlebars
      for (const file of await glob(`${site.outDir}/**/*.{ts,tsx,json}`)) {
        const template = Handlebars.compile(await readFile(file, 'utf-8'));
        await writeFile(file, template(site));
      }

      // 3. Rename pattern files
      // ...
    });
  }

  await queue.onIdle();
}
```

---

## Decision

**CONFIRMED: giget + Handlebars + Custom (hybrid approach)**

Rationale:
1. No single tool handles batch generation
2. giget is optimal for parallel template download
3. Handlebars provides safe, familiar templating
4. Custom orchestrator enables 10-100 site parallelism
5. Minimal dependencies, maximum control

---

## Sources

- [giget - UnJS Package](https://unjs.io/packages/giget/)
- [Plop.js Official](https://plopjs.com/)
- [Hygen Official](http://www.hygen.io/)
- [Handlebars.js](https://handlebarsjs.com/)
- [degit - Rich Harris](https://github.com/Rich-Harris/degit)
