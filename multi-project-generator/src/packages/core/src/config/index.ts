/**
 * Configuration module for MPG
 *
 * Handles loading and validating YAML configuration files using Zod.
 */

import { z } from 'zod/v4';
import type { MPGConfig, SiteConfig } from '@mpg/shared';

/**
 * Zod schema for site configuration
 */
export const SiteConfigSchema = z.object({
  id: z.string().min(1).max(50),
  name: z.string(),
  domain: z.string().url(),
  type: z.enum(['marketing', 'docs', 'app', 'blog', 'ecommerce']),
  framework: z.enum(['next', 'astro', 'remix', 'nuxt', 'svelte']),
  template: z.string().default('default'),
  workflow: z.string().default('standard'),
  priority: z.number().int().min(0).max(10).default(0),
  variables: z.record(z.unknown()).optional().default({}),
});

/**
 * Zod schema for full MPG configuration
 */
export const MPGConfigSchema = z.object({
  version: z.literal('1.0'),
  output: z.object({
    dir: z.string().default('./generated'),
    clean: z.boolean().default(false),
    monorepo: z.boolean().default(true),
  }),
  defaults: z.object({
    framework: z.enum(['next', 'astro', 'remix', 'nuxt', 'svelte']).default('next'),
    template: z.string().default('default'),
    workflow: z.string().default('standard'),
  }),
  workflows: z.record(z.string()).default({}),
  templates: z.record(z.object({
    source: z.string(),
    variables: z.record(z.unknown()).optional(),
  })).default({}),
  sites: z.array(SiteConfigSchema),
});

/**
 * Load configuration from a YAML file
 *
 * @param path - Path to the YAML configuration file
 * @returns Validated MPG configuration
 */
export async function loadConfig(_path: string): Promise<MPGConfig> {
  // TODO: Implement YAML loading and Zod validation
  throw new Error('Not implemented');
}

/**
 * Validate a configuration object
 *
 * @param config - Raw configuration object
 * @returns Validated MPG configuration
 */
export function validateConfig(config: unknown): MPGConfig {
  return MPGConfigSchema.parse(config) as MPGConfig;
}

/**
 * Get effective site config with defaults applied
 */
export function resolveSiteConfig(
  _site: Partial<SiteConfig>,
  _defaults: MPGConfig['defaults']
): SiteConfig {
  // TODO: Merge site config with defaults
  throw new Error('Not implemented');
}
