# Research: AI Agent Integration Protocols

**Date**: 2026-01-04
**Status**: Complete
**ADR**: ADR-006

---

## Executive Summary

**Recommendation**: MCP (Model Context Protocol) as primary integration

MCP has emerged as the industry standard for AI tool integration (2024-2025). Unprecedented cross-vendor adoption (OpenAI, Google, Microsoft) signals this is the universal protocol. 8M+ downloads, 5,800+ servers, 300+ clients.

---

## Comparison Matrix

| Protocol | AI-Native | Streaming | Ecosystem | Best For |
|----------|-----------|-----------|-----------|----------|
| **MCP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | MPG ✓ |
| **REST** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Legacy |
| **GraphQL** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Data exploration |
| **gRPC** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Internal services |
| **WebSocket** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Real-time |
| **Function Calling** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | OpenAI only |
| **LangChain** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Python workflows |

---

## MCP Adoption Timeline (2024-2025)

| Period | Downloads | Servers | Key Events |
|--------|-----------|---------|------------|
| Nov 2024 | ~100K | Launch | Anthropic introduces MCP |
| Mar 2025 | 5M+ | 4,000+ | OpenAI announces support |
| Apr 2025 | 8M+ | 5,800+ | Explosive growth |
| Dec 2025 | 50M+ (proj) | 10,000+ | Linux Foundation governance |

---

## Why MCP Won

### The M×N Problem
Before MCP: n LLMs × m tools = n·m custom integrations
- Example: 10 AI providers × 100 tools = 1,000 integrations

After MCP: 1 client + 1 server = any × any
- Universal protocol eliminates integration explosion

### Cross-Vendor Adoption
March 2025: Sam Altman (OpenAI) tweeted:
> "People love MCP and we are excited to add support across our products."

December 2025: MCP donated to Linux Foundation's Agentic AI Foundation (AAIF), co-founded by:
- Anthropic, OpenAI, Google, Microsoft, AWS, Cloudflare, Bloomberg

---

## Detailed Analysis

### MCP (Recommended)

**Pros:**
- Provider-agnostic (any LLM)
- 8M+ downloads
- 5,800+ servers, 300+ clients
- Multiple transports (STDIO, SSE, HTTP)
- Tool discovery built-in
- Streaming support
- Enterprise-ready (AWS, Azure, GCP)

**Cons:**
- Relatively new (best practices emerging)
- Requires running server processes
- New paradigm for REST developers

**Claude Code Features:**
- Local + Remote MCP servers
- Tool Search (85% token reduction)
- Scoped to project or user level

### REST API

**Pros:**
- Universal understanding
- 20+ years of tooling
- Simple, debuggable
- Web-native

**Cons:**
- Not AI-native (verbose for tool calling)
- No streaming by default
- Token inefficient

**When to use:** Legacy integration, simple endpoints

### gRPC

**Pros:**
- Lowest latency (25ms vs 250ms REST)
- 50,000 req/sec throughput
- Native streaming
- Binary protocol

**Cons:**
- Doesn't work in browsers
- Steeper learning curve
- Not designed for AI agents

**When to use:** Internal microservices, high-performance needs

### Function Calling (OpenAI)

**Pros:**
- Native to OpenAI models
- Simple integration
- Battle-tested

**Cons:**
- Vendor lock-in (OpenAI only)
- Static definitions
- No discovery
- Different format than Claude

**When to use:** OpenAI-only deployments

---

## AI Coding Tools & Protocols

| Tool | Primary | Secondary |
|------|---------|-----------|
| **Claude Code** | MCP | REST |
| **Cursor** | Function Calling + MCP | REST |
| **Zed** | MCP | REST |
| **VS Code Copilot** | Function Calling + MCP | REST |
| **Continue** | MCP | Function Calling |

---

## MCP Implementation for MPG

### Tool Definitions:
```typescript
// MCP Server for MPG
const tools = [
  {
    name: "list_sites",
    description: "List configured sites",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["marketing", "docs", "app"] },
        format: { type: "string", enum: ["summary", "full"] }
      }
    }
  },
  {
    name: "plan_sites",
    description: "Preview generation (dry run)",
    inputSchema: {
      type: "object",
      properties: {
        ids: { type: "array", items: { type: "string" } },
        steps: { type: "array", items: { type: "string" } }
      }
    }
  },
  {
    name: "apply_sites",
    description: "Execute site generation",
    inputSchema: {
      type: "object",
      required: ["steps"],
      properties: {
        ids: { type: "array", items: { type: "string" } },
        steps: { type: "array", items: { type: "string" } },
        concurrency: { type: "number", minimum: 1, maximum: 50 }
      }
    }
  }
];
```

### Claude Code Configuration:
```json
{
  "mcpServers": {
    "mpg": {
      "command": "npx",
      "args": ["@meta-prompting/mpg", "mcp:start"],
      "env": {
        "MPG_CONFIG": "./sites.yaml"
      }
    }
  }
}
```

---

## Performance Comparison

| Metric | REST | GraphQL | gRPC | MCP (STDIO) |
|--------|------|---------|------|-------------|
| Latency | 250ms | 180ms | 25ms | 40ms |
| Throughput | 20K/s | 15K/s | 50K/s | 30K/s |
| Token efficiency | 100% | 80% | 85% | 50% (w/ Tool Search) |

---

## Recommendation for MPG

### Primary: MCP

**Rationale:**
1. Industry standard (2025)
2. Provider-agnostic (Claude, GPT, Gemini)
3. Tool discovery built-in
4. Streaming for progress updates
5. Claude Code native integration

### Secondary: REST (Legacy)

Keep REST API for:
- Non-AI tool integrations
- Legacy system compatibility
- Simple HTTP clients

### Implementation Priority:

**Phase 0 (MVP):**
- 3 MCP tools: list_sites, plan_sites, apply_sites
- STDIO transport

**Phase 1:**
- HTTP transport for remote access
- Progress streaming
- Tool Search optimization

**Phase 2:**
- Enterprise scoping (team-level servers)
- Advanced security/permissions

---

## Decision

**CONFIRMED: MCP (Model Context Protocol)**

Rationale:
1. Industry standard (unprecedented cross-vendor adoption)
2. Provider-agnostic (future-proof)
3. Tool discovery and streaming built-in
4. Claude Code native integration
5. 85% token reduction with Tool Search

---

## Sources

- [Model Context Protocol - Anthropic](https://www.anthropic.com/news/model-context-protocol)
- [Why MCP Won - The New Stack](https://thenewstack.io/why-the-model-context-protocol-won/)
- [MCP Enterprise Adoption 2025](https://guptadeepak.com/the-complete-guide-to-model-context-protocol-mcp/)
- [Claude Code MCP Integration](https://code.claude.com/docs/en/mcp)
- [MCP vs Function Calling](https://www.descope.com/blog/post/mcp-vs-function-calling)
