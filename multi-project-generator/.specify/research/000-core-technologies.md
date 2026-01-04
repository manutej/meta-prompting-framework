# Core Technologies Research (High Priority)

**Date**: 2026-01-04
**Status**: Complete
**Source**: Context7 + Web Research (Parallel Agents)
**Priority**: HIGH - Foundation for all ADRs

---

## Executive Summary

This document consolidates up-to-date documentation for all core MPG technologies. All versions and APIs verified against latest 2025-2026 releases.

| Technology | Version | Last Updated | Status |
|------------|---------|--------------|--------|
| **Turborepo** | 2.7.0 | Dec 2025 | ✅ Production Ready |
| **Zod** | 4.3.5 | Jul 2025 | ✅ Production Ready |
| **p-queue** | 9.0.1 | Dec 2025 | ✅ Production Ready |
| **Commander.js** | 14.0.2 | Nov 2024 | ✅ Production Ready |
| **giget** | 2.0.0 | Feb 2025 | ✅ Production Ready |
| **Handlebars** | 4.7.8 | Stable | ✅ Production Ready |
| **MCP SDK** | 1.25.1 | Dec 2025 | ✅ Production Ready |

---

## 1. Turborepo 2.7.0 (ADR-001)

### What's New in 2025

**Rust Rewrite Complete**: Turborepo is now 100% Rust (replacing Go), providing:
- 3x faster builds than Nx
- Improved reliability
- Better memory efficiency

**New Features (v2.7)**:
- **Devtools**: Visual Package/Task Graph explorer (`turbo devtools`)
- **Composable Configuration**: Package configs can extend others
- **Yarn 4 Catalogs Support**
- **Biome Integration**: Environment variable linting

### Configuration for MPG

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"],
      "env": ["NODE_ENV"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  },
  "globalDependencies": [".env", "tsconfig.json"]
}
```

### Key Requirements
- **Node.js**: 18+ (20+ recommended)
- **Package Manager**: pnpm 9.x recommended
- **Caching**: Free Vercel Remote Cache available

### Breaking Changes from v1.x
- `outputMode` → `outputLogs`
- Strict environment mode is default
- `packageManager` field required in package.json

---

## 2. Zod 4.3.5 (ADR-002)

### What's New in v4

**Performance Revolution**:
- 14x faster string parsing
- 7x faster array parsing
- 6.5x faster object parsing
- 57% smaller bundle (10KB vs 20KB)

**New Features**:
- `z.email()`, `z.uuid()`, `z.url()` top-level functions
- `z.stringbool()` for string-to-boolean conversion
- `z.file()` for File validation
- `z.interface()` replaces `z.lazy()` for recursion
- Native `.toJSONSchema()` method
- **Standard Schema 1.0** compatibility

### Import Pattern
```typescript
// v4 subpath import
import { z } from 'zod/v4';

// Can run v3 and v4 side-by-side during migration
import { z as z3 } from 'zod/v3';
import { z as z4 } from 'zod/v4';
```

### YAML Validation Example
```typescript
import { z } from 'zod/v4';

const siteConfigSchema = z.object({
  id: z.string(),
  type: z.enum(['marketing', 'docs', 'app']),
  domain: z.string().url(),
  template: z.string(),
  variables: z.record(z.string(), z.unknown()).optional()
});

// Infer TypeScript type
type SiteConfig = z.infer<typeof siteConfigSchema>;

// Validate with pretty errors
try {
  const config = siteConfigSchema.parse(yamlData);
} catch (error) {
  console.error(z.prettifyError(error));
}
```

### Migration from v3
```bash
npx zod-v3-to-v4 ./src
```

---

## 3. p-queue 9.0.1 (ADR-003)

### Critical Update: ESM Only
**p-queue v9.x is ESM-only**. Use `p-queue-compat` if CommonJS required.

### Features Confirmed
| Feature | Status |
|---------|--------|
| Priority Queue | ✅ |
| Pause/Resume | ✅ |
| Rate Limiting (intervalCap) | ✅ |
| onIdle() / onEmpty() | ✅ |
| Timeout Support | ✅ |
| AbortController | ✅ |

### Batch Site Generation Pattern
```typescript
import PQueue from 'p-queue';
import pRetry from 'p-retry';

const queue = new PQueue({
  concurrency: 20,           // 20 sites simultaneously
  intervalCap: 50,           // Max 50 per minute
  interval: 60000,           // 1 minute window
  timeout: 300000            // 5 min timeout per site
});

async function generateSites(sites: SiteConfig[]) {
  const results = await Promise.all(
    sites.map(site =>
      queue.add(
        async ({ signal }) => pRetry(
          () => buildSite(site, signal),
          { retries: 3, factor: 2 }
        ),
        { priority: site.priority || 0 }
      ).catch(error => ({ site: site.id, error }))
    )
  );

  await queue.onIdle();
  return results;
}
```

### Breaking Changes (v9.0.0)
- `throwOnTimeout` removed - always throws `TimeoutError`
- Node.js 20+ required
- ESM-only (no CommonJS)

---

## 4. Commander.js 14.0.2 (ADR-004)

### Key Features for MPG
- **Subcommand patterns**: Action handlers or standalone executables
- **Option grouping**: Organized help output
- **TypeScript**: Enhanced types via `@commander-js/extra-typings`
- **Async support**: `.parseAsync()` for async handlers

### TypeScript Setup
```typescript
import { Command } from '@commander-js/extra-typings';

const program = new Command()
  .name('mpg')
  .version('1.0.0')
  .option('-v, --verbose', 'enable verbose logging')
  .option('-c, --config <path>', 'config file path');

// Subcommands
program
  .command('list')
  .description('List all configured sites')
  .option('-f, --format <type>', 'output format', 'table')
  .action(async (options) => {
    // Fully typed options
  });

program
  .command('plan')
  .argument('[sites...]', 'specific sites to plan')
  .option('--dry-run', 'preview without execution')
  .action(async (sites, options) => {
    // sites: string[], options: typed
  });

program
  .command('apply')
  .option('--force', 'skip confirmation')
  .option('--parallel <n>', 'concurrent jobs', parseInt, 10)
  .action(async (options) => {
    // Execute site generation
  });

await program.parseAsync(process.argv);
```

### Requirements
- Node.js 20+
- TypeScript 5.x for enhanced typings

---

## 5. giget 2.0.0 + Handlebars 4.7.8 (ADR-005)

### giget: Template Downloading

```typescript
import { downloadTemplate } from 'giget';

// Basic usage
const { dir } = await downloadTemplate('github:org/template');

// With options
await downloadTemplate('org/template#main', {
  dir: './output',
  force: 'clean',        // Clean directory first
  preferOffline: true,   // Use cache when available
  auth: process.env.GITHUB_TOKEN  // For private repos
});
```

**Supported Sources**:
- GitHub: `github:user/repo` or `gh:user/repo` or `user/repo`
- GitLab: `gitlab:user/repo`
- Bitbucket: `bitbucket:user/repo`

### Handlebars: Variable Substitution

```typescript
import Handlebars from 'handlebars';

// Compile template
const template = Handlebars.compile('{{siteName}} - {{description}}');

// Render with data
const output = template({
  siteName: 'My Site',
  description: 'A great website'
});

// Register custom helpers
Handlebars.registerHelper('uppercase', (str) => str.toUpperCase());

// Security: Always escape user input
Handlebars.registerHelper('safeLink', (url, text) => {
  const escapedUrl = Handlebars.Utils.escapeExpression(url);
  const escapedText = Handlebars.Utils.escapeExpression(text);
  return new Handlebars.SafeString(`<a href="${escapedUrl}">${escapedText}</a>`);
});
```

### Security Best Practices
- Always use `{{variable}}` (auto-escapes HTML)
- Never use `{{{raw}}}` with user input
- Use `Handlebars.Utils.escapeExpression()` in helpers
- Keep Handlebars updated (CVE-2021 patched in 4.7.7+)

---

## 6. MCP SDK 1.25.1 (ADR-006)

### Current State (Dec 2025)
- **5,800+ MCP servers** in ecosystem
- **300+ MCP clients** deployed
- Donated to **Linux Foundation** (Agentic AI Foundation)
- OpenAI, Google, Microsoft all adopted MCP

### Server Implementation

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ErrorCode
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "mpg-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// List tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "list_sites",
      description: "List all configured sites from YAML",
      inputSchema: {
        type: "object",
        properties: {
          format: { type: "string", enum: ["summary", "full"] }
        }
      }
    },
    {
      name: "plan_sites",
      description: "Generate execution plan for site generation",
      inputSchema: {
        type: "object",
        properties: {
          sites: { type: "array", items: { type: "string" } },
          dryRun: { type: "boolean" }
        }
      }
    },
    {
      name: "apply_sites",
      description: "Execute site generation plan",
      inputSchema: {
        type: "object",
        properties: {
          concurrency: { type: "number", default: 10 },
          force: { type: "boolean" }
        }
      }
    }
  ]
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "list_sites":
      return { content: [{ type: "text", text: JSON.stringify(sites) }] };
    case "plan_sites":
      return { content: [{ type: "text", text: JSON.stringify(plan) }] };
    case "apply_sites":
      return { content: [{ type: "text", text: "Generation complete" }] };
    default:
      throw new McpError(ErrorCode.ToolNotFound, `Tool ${name} not found`);
  }
});

// Connect
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Transport Options
| Transport | Use Case | Command |
|-----------|----------|---------|
| **stdio** | Local tools | `claude mcp add --transport stdio mpg node dist/server.js` |
| **HTTP** | Remote/Production | `claude mcp add --transport http mpg https://api.example.com/mcp` |
| **SSE** | Legacy only | Deprecated - use HTTP |

### Testing with MCP Inspector
```bash
npx @modelcontextprotocol/inspector node dist/server.js
# Opens UI at http://localhost:6274
```

---

## Technology Stack Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                        MPG CLI (Commander.js 14.0.2)           │
│                     /mpg list | plan | apply | status          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐    ┌──────────────────────────────────┐ │
│  │ Config Loader    │    │ MCP Server (@mcp/sdk 1.25.1)     │ │
│  │ YAML → Zod 4.3.5 │    │ list_sites | plan_sites | apply  │ │
│  └──────────────────┘    └──────────────────────────────────┘ │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐    ┌──────────────────────────────────┐ │
│  │ Template Engine  │    │ Task Queue (p-queue 9.0.1)       │ │
│  │ giget 2.0.0 +    │───▶│ • 20 concurrent sites            │ │
│  │ Handlebars 4.7.8 │    │ • Priority-based                 │ │
│  └──────────────────┘    │ • Pause/resume                   │ │
│                          └──────────────────────────────────┘ │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│              Output: Turborepo 2.7.0 Monorepo                  │
│                    pnpm workspaces + Remote Cache              │
└────────────────────────────────────────────────────────────────┘
```

---

## Version Compatibility Matrix

| Dependency | Min Version | Recommended | Notes |
|------------|-------------|-------------|-------|
| Node.js | 18.x | 20.x+ | p-queue 9.x requires 20+ |
| TypeScript | 5.0 | 5.3+ | For enhanced Commander types |
| pnpm | 8.x | 9.1.0 | Turborepo workspace support |
| Zod | 3.25 | 4.x (via v4 import) | MCP SDK peer dependency |

---

## Migration Considerations

### From Earlier Versions

1. **p-limit → p-queue**: Similar API, add priority support
2. **Zod v3 → v4**: Use subpath import, run codemod
3. **Turborepo v1 → v2**: Run `@turbo/codemod migrate`
4. **MCP v1.x → v2**: Wait for v2 release (Q1 2026)

### Package.json Template

```json
{
  "name": "mpg",
  "version": "1.0.0",
  "type": "module",
  "packageManager": "pnpm@9.1.0",
  "engines": {
    "node": ">=20.0.0"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.25.1",
    "commander": "^14.0.2",
    "giget": "^2.0.0",
    "handlebars": "^4.7.8",
    "p-queue": "^9.0.1",
    "p-retry": "^6.2.0",
    "yaml": "^2.3.4",
    "zod": "^3.25.0"
  },
  "devDependencies": {
    "@commander-js/extra-typings": "^14.0.0",
    "@types/node": "^20.0.0",
    "turbo": "^2.7.0",
    "typescript": "^5.3.0"
  }
}
```

---

## Sources

### Turborepo
- [Turborepo 2.7 Release](https://turborepo.com/blog/turbo-2-7)
- [Turborepo Documentation](https://turborepo.com/docs)
- [Configuration Reference](https://turborepo.com/docs/reference/configuration)

### Zod
- [Zod v4 Release Notes](https://zod.dev/v4)
- [Migration Guide](https://zod.dev/v4/changelog)
- [Standard Schema 1.0](https://x.com/colinhacks/status/1883907825384190418)

### p-queue
- [p-queue npm](https://www.npmjs.com/package/p-queue)
- [GitHub sindresorhus/p-queue](https://github.com/sindresorhus/p-queue)
- [p-retry integration](https://github.com/sindresorhus/p-retry)

### Commander.js
- [Commander.js npm](https://www.npmjs.com/package/commander)
- [GitHub tj/commander.js](https://github.com/tj/commander.js)
- [@commander-js/extra-typings](https://github.com/commander-js/extra-typings)

### giget + Handlebars
- [giget npm](https://www.npmjs.com/package/giget)
- [GitHub unjs/giget](https://github.com/unjs/giget)
- [Handlebars Documentation](https://handlebarsjs.com/)

### MCP SDK
- [MCP SDK npm](https://www.npmjs.com/package/@modelcontextprotocol/sdk)
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)
- [One Year of MCP](http://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)
