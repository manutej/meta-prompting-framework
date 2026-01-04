/**
 * Apply sites tool handler
 */

interface ApplySitesArgs {
  concurrency?: number;
  sites?: string[];
  force?: boolean;
}

export async function applySites(args: ApplySitesArgs): Promise<string> {
  const { concurrency = 10, sites, force = false } = args;

  // TODO: Execute site generation
  const response = {
    status: 'not_implemented',
    concurrency,
    sites: sites || 'all',
    force,
  };

  return JSON.stringify(response, null, 2);
}
