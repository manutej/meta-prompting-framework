/**
 * Shared TypeScript types for MPG
 *
 * Core interfaces used across all packages.
 */

/**
 * Site type enumeration
 */
export type SiteType = 'marketing' | 'docs' | 'app' | 'blog' | 'ecommerce';

/**
 * Framework options for generated sites
 */
export type Framework = 'next' | 'astro' | 'remix' | 'nuxt' | 'svelte';

/**
 * Individual site configuration
 */
export interface SiteConfig {
  id: string;
  name: string;
  domain: string;
  type: SiteType;
  framework: Framework;
  template: string;
  workflow: string;
  priority: number;
  variables: Record<string, unknown>;
}

/**
 * Full MPG configuration from YAML
 */
export interface MPGConfig {
  version: '1.0';
  output: OutputConfig;
  defaults: DefaultsConfig;
  workflows: Record<string, string>;
  templates: Record<string, TemplateConfig>;
  sites: SiteConfig[];
}

/**
 * Output configuration
 */
export interface OutputConfig {
  dir: string;
  clean: boolean;
  monorepo: boolean;
}

/**
 * Default values for sites
 */
export interface DefaultsConfig {
  framework: Framework;
  template: string;
  workflow: string;
}

/**
 * Template configuration
 */
export interface TemplateConfig {
  source: string;
  variables?: Record<string, unknown>;
}

/**
 * Execution plan for site generation
 */
export interface ExecutionPlan {
  sites: PlannedSite[];
  totalEstimatedTime: number;
  warnings: string[];
}

/**
 * Individual site in execution plan
 */
export interface PlannedSite {
  config: SiteConfig;
  steps: ExecutionStep[];
  estimatedTime: number;
}

/**
 * Single execution step
 */
export interface ExecutionStep {
  name: string;
  action: string;
  dependencies: string[];
}

/**
 * Result of site generation
 */
export interface GenerationResult {
  siteId: string;
  success: boolean;
  outputPath?: string;
  error?: Error;
  duration: number;
}
