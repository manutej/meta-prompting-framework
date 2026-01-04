/**
 * Error classes for MPG
 *
 * Custom error types for different failure scenarios.
 */

/**
 * Base error class for all MPG errors
 */
export class MPGError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = 'MPGError';
  }
}

/**
 * Error thrown when configuration validation fails
 */
export class ValidationError extends MPGError {
  constructor(
    message: string,
    public readonly path?: string,
    cause?: Error
  ) {
    super(message, 'VALIDATION_ERROR', cause);
    this.name = 'ValidationError';
  }
}

/**
 * Error thrown when site generation fails
 */
export class ExecutionError extends MPGError {
  constructor(
    message: string,
    public readonly siteId?: string,
    cause?: Error
  ) {
    super(message, 'EXECUTION_ERROR', cause);
    this.name = 'ExecutionError';
  }
}

/**
 * Error thrown when template operations fail
 */
export class TemplateError extends MPGError {
  constructor(
    message: string,
    public readonly templateId?: string,
    cause?: Error
  ) {
    super(message, 'TEMPLATE_ERROR', cause);
    this.name = 'TemplateError';
  }
}
