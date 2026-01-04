/**
 * @mpg/templates - Template handling for Multi-Project Generator
 *
 * This package provides template functionality:
 * - Download: Fetch templates from GitHub/GitLab using giget
 * - Process: Variable substitution with Handlebars
 * - Validate: Schema and structure validation
 * - Registry: Local and remote template management
 */

export * from './download/index.js';
export * from './process/index.js';
export * from './validate/index.js';
export * from './registry/index.js';
