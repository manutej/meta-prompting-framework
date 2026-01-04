#!/usr/bin/env node
/**
 * @mpg/cli - Multi-Project Generator CLI
 *
 * Command-line interface for generating multiple sites from YAML configuration.
 */

import { Command } from '@commander-js/extra-typings';
import { listCommand } from './commands/list.js';
import { planCommand } from './commands/plan.js';
import { applyCommand } from './commands/apply.js';
import { statusCommand } from './commands/status.js';
import { validateCommand } from './commands/validate.js';

const program = new Command()
  .name('mpg')
  .description('Multi-Project Generator - Generate multiple sites from YAML configuration')
  .version('0.1.0')
  .option('-v, --verbose', 'Enable verbose output', false)
  .option('-c, --config <path>', 'Path to configuration file', 'sites.yaml');

// Register commands
program.addCommand(listCommand);
program.addCommand(planCommand);
program.addCommand(applyCommand);
program.addCommand(statusCommand);
program.addCommand(validateCommand);

// Parse and execute
program.parseAsync(process.argv).catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
