/**
 * @mpg/core - Core engine for Multi-Project Generator
 *
 * This package provides the main functionality:
 * - Config: Load and validate YAML configuration with Zod
 * - Plan: Generate execution plans with dependency resolution
 * - Execute: Run site generation with p-queue concurrency
 * - Output: Write to Turborepo monorepo structure
 */

export * from './config/index.js';
export * from './plan/index.js';
export * from './execute/index.js';
export * from './output/index.js';
