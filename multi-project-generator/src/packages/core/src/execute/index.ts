/**
 * Execution module for MPG
 *
 * Runs site generation with p-queue for concurrency control.
 */

import PQueue from 'p-queue';
import type { ExecutionPlan, GenerationResult } from '@mpg/shared';

/**
 * Execution options
 */
export interface ExecuteOptions {
  concurrency?: number;
  dryRun?: boolean;
  onProgress?: (result: GenerationResult) => void;
}

/**
 * Execute a generation plan
 *
 * @param plan - Execution plan to run
 * @param options - Execution options
 * @returns Array of generation results
 */
export async function executePlan(
  plan: ExecutionPlan,
  options: ExecuteOptions = {}
): Promise<GenerationResult[]> {
  const { concurrency = 10, dryRun = false, onProgress } = options;

  const queue = new PQueue({
    concurrency,
    timeout: 300000, // 5 minutes per site
  });

  const results: GenerationResult[] = [];

  for (const site of plan.sites) {
    queue.add(
      async () => {
        const startTime = Date.now();
        let result: GenerationResult;

        try {
          if (dryRun) {
            // Simulate generation
            result = {
              siteId: site.config.id,
              success: true,
              duration: 0,
            };
          } else {
            // TODO: Actual generation
            result = {
              siteId: site.config.id,
              success: true,
              duration: Date.now() - startTime,
            };
          }
        } catch (error) {
          result = {
            siteId: site.config.id,
            success: false,
            error: error as Error,
            duration: Date.now() - startTime,
          };
        }

        results.push(result);
        onProgress?.(result);
        return result;
      },
      { priority: site.config.priority }
    );
  }

  await queue.onIdle();
  return results;
}

/**
 * Execute a single site generation
 */
export async function executeSite(
  _siteId: string,
  _plan: ExecutionPlan
): Promise<GenerationResult> {
  // TODO: Implement single site execution
  throw new Error('Not implemented');
}

/**
 * Pause execution
 */
export function pauseExecution(_queue: PQueue): void {
  _queue.pause();
}

/**
 * Resume execution
 */
export function resumeExecution(_queue: PQueue): void {
  _queue.start();
}
