/**
 * Apply command - Execute site generation
 */

import { Command } from '@commander-js/extra-typings';

function parseIntOption(value: string): number {
  const parsed = parseInt(value, 10);
  if (isNaN(parsed)) {
    throw new Error('Not a valid number');
  }
  return parsed;
}

export const applyCommand = new Command('apply')
  .description('Execute site generation')
  .argument('[plan-file]', 'Path to plan file to execute')
  .option('--force', 'Force execution even with conflicts', false)
  .option('--skip-confirm', 'Skip confirmation prompt', false)
  .option('--concurrency <n>', 'Number of parallel jobs', parseIntOption, 10)
  .action(async (planFile, options) => {
    console.log('Executing site generation...');

    if (planFile) {
      console.log('Using plan file:', planFile);
    }

    console.log('Concurrency:', options.concurrency);

    if (options.force) {
      console.log('[FORCE] Ignoring conflicts');
    }

    // TODO: Execute generation
    console.log('(Not implemented yet)');
  });
