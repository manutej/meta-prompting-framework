/**
 * Template validation module
 *
 * Validates template structure and metadata.
 */

import { z } from 'zod/v4';
import type { TemplateError } from '@mpg/shared';

/**
 * Template metadata schema
 */
export const TemplateMetadataSchema = z.object({
  name: z.string(),
  version: z.string().optional(),
  description: z.string().optional(),
  framework: z.enum(['next', 'astro', 'remix', 'nuxt', 'svelte']).optional(),
  variables: z.record(z.object({
    type: z.enum(['string', 'number', 'boolean']),
    required: z.boolean().default(false),
    default: z.unknown().optional(),
    description: z.string().optional(),
  })).optional(),
});

/**
 * Template metadata type
 */
export type TemplateMetadata = z.infer<typeof TemplateMetadataSchema>;

/**
 * Validate a template directory
 *
 * @param dir - Template directory path
 * @returns Validation result
 */
export async function validateTemplate(
  _dir: string
): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];

  // TODO: Check for required files
  // TODO: Validate template.json metadata
  // TODO: Check for valid variable placeholders

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate template variables against metadata
 */
export function validateVariables(
  variables: Record<string, unknown>,
  metadata: TemplateMetadata
): string[] {
  const errors: string[] = [];

  if (metadata.variables) {
    for (const [name, schema] of Object.entries(metadata.variables)) {
      const value = variables[name];

      if (schema.required && value === undefined) {
        errors.push(`Missing required variable: ${name}`);
      }

      if (value !== undefined) {
        const expectedType = schema.type;
        const actualType = typeof value;

        if (actualType !== expectedType) {
          errors.push(
            `Variable ${name}: expected ${expectedType}, got ${actualType}`
          );
        }
      }
    }
  }

  return errors;
}

/**
 * Load template metadata from template.json
 */
export async function loadTemplateMetadata(
  _dir: string
): Promise<TemplateMetadata | null> {
  // TODO: Read and parse template.json
  return null;
}
