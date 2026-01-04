/**
 * Template processing module
 *
 * Uses Handlebars for variable substitution in template files.
 */

import Handlebars from 'handlebars';
import { glob } from 'glob';
import { readFile, writeFile } from 'fs/promises';

/**
 * Variables for template substitution
 */
export type TemplateVariables = Record<string, unknown>;

/**
 * Process options
 */
export interface ProcessOptions {
  patterns?: string[];
  exclude?: string[];
}

/**
 * Process all template files in a directory
 *
 * @param dir - Directory containing template files
 * @param variables - Variables to substitute
 * @param options - Processing options
 */
export async function processTemplateDir(
  dir: string,
  variables: TemplateVariables,
  options: ProcessOptions = {}
): Promise<void> {
  const patterns = options.patterns || ['**/*.{ts,tsx,js,jsx,json,md,yaml,yml}'];
  const exclude = options.exclude || ['**/node_modules/**', '**/dist/**'];

  const files = await glob(patterns, {
    cwd: dir,
    absolute: true,
    ignore: exclude,
  });

  await Promise.all(
    files.map(async (file) => {
      const content = await readFile(file, 'utf-8');
      const processed = processTemplate(content, variables);
      await writeFile(file, processed, 'utf-8');
    })
  );
}

/**
 * Process a single template string
 *
 * @param template - Template string with {{variable}} placeholders
 * @param variables - Variables to substitute
 * @returns Processed string
 */
export function processTemplate(
  template: string,
  variables: TemplateVariables
): string {
  const compiled = Handlebars.compile(template);
  return compiled(variables);
}

/**
 * Rename files with __pattern__ in their names
 *
 * @param dir - Directory to process
 * @param variables - Variables for substitution
 */
export async function renameTemplateFiles(
  _dir: string,
  _variables: TemplateVariables
): Promise<void> {
  // TODO: Find files with __pattern__ and rename them
  // e.g., __name__.tsx -> {variables.name}.tsx
}

/**
 * Register a custom Handlebars helper
 */
export function registerHelper(
  name: string,
  fn: Handlebars.HelperDelegate
): void {
  Handlebars.registerHelper(name, fn);
}

// Register default helpers
registerHelper('uppercase', (str: string) => str.toUpperCase());
registerHelper('lowercase', (str: string) => str.toLowerCase());
registerHelper('capitalize', (str: string) =>
  str.charAt(0).toUpperCase() + str.slice(1)
);
