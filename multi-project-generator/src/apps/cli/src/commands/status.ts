/**
 * Status command - Show generation status
 */

import { Command } from '@commander-js/extra-typings';

export const statusCommand = new Command('status')
  .description('Show current generation status')
  .option('-d, --detailed', 'Show detailed status', false)
  .option('--show-errors', 'Include error details', false)
  .action(async (options) => {
    console.log('Checking status...');

    if (options.detailed) {
      console.log('Showing detailed status');
    }

    // TODO: Check and display status
    console.log('(Not implemented yet)');
  });
