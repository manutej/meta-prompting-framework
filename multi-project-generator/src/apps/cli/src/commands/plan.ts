/**
 * Plan command - Generate execution plan
 */

import { Command } from '@commander-js/extra-typings';

export const planCommand = new Command('plan')
  .description('Generate and display execution plan')
  .argument('[sites...]', 'Specific sites to plan')
  .option('--dry-run', 'Preview without any side effects', false)
  .option('-o, --output <file>', 'Save plan to file')
  .action(async (sites, options) => {
    console.log('Generating execution plan...');

    if (sites.length > 0) {
      console.log('Sites:', sites.join(', '));
    } else {
      console.log('Planning all sites');
    }

    if (options.dryRun) {
      console.log('[DRY RUN] No changes will be made');
    }

    // TODO: Load config and generate plan
    console.log('(Not implemented yet)');
  });
