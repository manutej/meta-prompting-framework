/**
 * MCP Tool definitions and handlers
 */

import type { Tool } from '@modelcontextprotocol/sdk/types.js';
import { listSites } from './list-sites.js';
import { planSites } from './plan-sites.js';
import { applySites } from './apply-sites.js';
import { getStatus } from './get-status.js';
import { validateConfig } from './validate-config.js';

/**
 * Register all available tools
 */
export function registerTools(): Tool[] {
  return [
    {
      name: 'list_sites',
      description: 'List all configured sites from the YAML configuration',
      inputSchema: {
        type: 'object',
        properties: {
          type: {
            type: 'string',
            enum: ['marketing', 'docs', 'app', 'blog', 'ecommerce'],
            description: 'Filter by site type',
          },
          format: {
            type: 'string',
            enum: ['summary', 'full'],
            description: 'Output format',
          },
        },
      },
    },
    {
      name: 'plan_sites',
      description: 'Generate an execution plan for site generation',
      inputSchema: {
        type: 'object',
        properties: {
          sites: {
            type: 'array',
            items: { type: 'string' },
            description: 'Specific sites to plan (optional)',
          },
          dryRun: {
            type: 'boolean',
            description: 'Preview only, no side effects',
          },
        },
      },
    },
    {
      name: 'apply_sites',
      description: 'Execute site generation based on configuration',
      inputSchema: {
        type: 'object',
        properties: {
          concurrency: {
            type: 'number',
            description: 'Number of parallel generation jobs',
            default: 10,
          },
          sites: {
            type: 'array',
            items: { type: 'string' },
            description: 'Specific sites to generate (optional)',
          },
          force: {
            type: 'boolean',
            description: 'Force generation even with conflicts',
          },
        },
      },
    },
    {
      name: 'get_status',
      description: 'Get the current status of site generation',
      inputSchema: {
        type: 'object',
        properties: {
          detailed: {
            type: 'boolean',
            description: 'Include detailed status per site',
          },
        },
      },
    },
    {
      name: 'validate_config',
      description: 'Validate the YAML configuration file',
      inputSchema: {
        type: 'object',
        properties: {
          configPath: {
            type: 'string',
            description: 'Path to configuration file',
          },
          strict: {
            type: 'boolean',
            description: 'Enable strict validation mode',
          },
        },
      },
    },
  ];
}

/**
 * Handle a tool call
 */
export async function handleToolCall(
  name: string,
  args: Record<string, unknown> = {}
): Promise<{ content: Array<{ type: string; text: string }> }> {
  let result: string;

  switch (name) {
    case 'list_sites':
      result = await listSites(args);
      break;
    case 'plan_sites':
      result = await planSites(args);
      break;
    case 'apply_sites':
      result = await applySites(args);
      break;
    case 'get_status':
      result = await getStatus(args);
      break;
    case 'validate_config':
      result = await validateConfig(args);
      break;
    default:
      throw new Error(`Unknown tool: ${name}`);
  }

  return {
    content: [
      {
        type: 'text',
        text: result,
      },
    ],
  };
}
