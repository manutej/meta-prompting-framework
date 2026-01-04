/**
 * Template registry module
 *
 * Manages local and remote template sources.
 */

/**
 * Template source definition
 */
export interface TemplateSource {
  id: string;
  name: string;
  source: string;
  description?: string;
  framework?: string;
}

/**
 * Registry options
 */
export interface RegistryOptions {
  cacheDir?: string;
}

/**
 * Template registry for managing available templates
 */
export class TemplateRegistry {
  private templates: Map<string, TemplateSource> = new Map();

  constructor(_options: RegistryOptions = {}) {
    // TODO: Initialize with cache directory
  }

  /**
   * Register a template source
   */
  register(template: TemplateSource): void {
    this.templates.set(template.id, template);
  }

  /**
   * Get a template by ID
   */
  get(id: string): TemplateSource | undefined {
    return this.templates.get(id);
  }

  /**
   * List all registered templates
   */
  list(): TemplateSource[] {
    return Array.from(this.templates.values());
  }

  /**
   * Filter templates by framework
   */
  filterByFramework(framework: string): TemplateSource[] {
    return this.list().filter((t) => t.framework === framework);
  }

  /**
   * Load templates from a remote registry URL
   */
  async loadFromRemote(_url: string): Promise<void> {
    // TODO: Fetch and parse remote registry
  }

  /**
   * Load templates from a local directory
   */
  async loadFromLocal(_dir: string): Promise<void> {
    // TODO: Scan local directory for templates
  }
}

/**
 * Default template registry instance
 */
export const defaultRegistry = new TemplateRegistry();

/**
 * Built-in templates
 */
export const BUILTIN_TEMPLATES: TemplateSource[] = [
  {
    id: 'next-marketing',
    name: 'Next.js Marketing',
    source: 'github:mpg-templates/next-marketing',
    description: 'Marketing site template with Next.js',
    framework: 'next',
  },
  {
    id: 'astro-docs',
    name: 'Astro Documentation',
    source: 'github:mpg-templates/astro-docs',
    description: 'Documentation site template with Astro',
    framework: 'astro',
  },
];
