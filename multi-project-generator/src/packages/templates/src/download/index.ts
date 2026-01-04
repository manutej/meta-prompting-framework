/**
 * Template download module
 *
 * Uses giget for fast parallel template downloads from GitHub/GitLab.
 */

import { downloadTemplate as gigetDownload } from 'giget';

/**
 * Download options
 */
export interface DownloadOptions {
  dir?: string;
  force?: boolean | 'clean';
  preferOffline?: boolean;
  auth?: string;
}

/**
 * Download result
 */
export interface DownloadResult {
  source: string;
  dir: string;
}

/**
 * Download a template from a remote source
 *
 * @param source - Template source (e.g., "github:org/repo" or "org/repo#branch")
 * @param options - Download options
 * @returns Download result with local directory
 */
export async function downloadTemplate(
  source: string,
  options: DownloadOptions = {}
): Promise<DownloadResult> {
  const result = await gigetDownload(source, {
    dir: options.dir,
    force: options.force,
    preferOffline: options.preferOffline,
    auth: options.auth || process.env['GIGET_AUTH'],
  });

  return {
    source: result.source,
    dir: result.dir,
  };
}

/**
 * Download multiple templates in parallel
 */
export async function downloadTemplates(
  sources: Array<{ source: string; options?: DownloadOptions }>
): Promise<DownloadResult[]> {
  return Promise.all(
    sources.map(({ source, options }) => downloadTemplate(source, options))
  );
}

/**
 * Check if a template is cached locally
 */
export async function isTemplateCached(_source: string): Promise<boolean> {
  // TODO: Check giget cache
  return false;
}
