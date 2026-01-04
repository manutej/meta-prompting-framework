/**
 * Planning module for MPG
 *
 * Generates execution plans with dependency resolution and resource estimation.
 */

import type { MPGConfig, ExecutionPlan, PlannedSite } from '@mpg/shared';

/**
 * Generate an execution plan from configuration
 *
 * @param config - Validated MPG configuration
 * @returns Execution plan with ordered sites and estimated times
 */
export async function generatePlan(_config: MPGConfig): Promise<ExecutionPlan> {
  // TODO: Implement plan generation
  // 1. Parse workflows for each site
  // 2. Resolve template dependencies
  // 3. Calculate execution order (priority-based)
  // 4. Estimate time for each site
  throw new Error('Not implemented');
}

/**
 * Plan a single site's generation
 */
export function planSite(
  _config: MPGConfig,
  _siteId: string
): PlannedSite {
  // TODO: Implement single site planning
  throw new Error('Not implemented');
}

/**
 * Validate that a plan is executable
 */
export function validatePlan(_plan: ExecutionPlan): string[] {
  // TODO: Check for circular dependencies, missing templates, etc.
  return [];
}

/**
 * Parse workflow string (e.g., "scaffold+design+deploy")
 */
export function parseWorkflow(workflow: string): string[] {
  return workflow.split('+').map((step) => step.trim());
}
