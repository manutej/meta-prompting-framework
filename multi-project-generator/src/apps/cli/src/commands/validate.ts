/**
 * Validate command - Validate configuration
 */

import { Command } from '@commander-js/extra-typings';

export const validateCommand = new Command('validate')
  .description('Validate configuration file')
  .argument('[config]', 'Path to configuration file', 'sites.yaml')
  .option('--strict', 'Enable strict validation', false)
  .action(async (configPath, options) => {
    console.log('Validating configuration:', configPath);

    if (options.strict) {
      console.log('Strict mode enabled');
    }

    // TODO: Load and validate config
    console.log('(Not implemented yet)');
  });
