/**
 * List command - Display configured sites
 */

import { Command } from '@commander-js/extra-typings';

export const listCommand = new Command('list')
  .description('List all configured sites')
  .option('-f, --format <type>', 'Output format (table, json, yaml)', 'table')
  .option('--type <type>', 'Filter by site type')
  .action(async (options) => {
    console.log('Listing sites...');
    console.log('Format:', options.format);

    if (options.type) {
      console.log('Filtering by type:', options.type);
    }

    // TODO: Load config and list sites
    console.log('(Not implemented yet)');
  });
