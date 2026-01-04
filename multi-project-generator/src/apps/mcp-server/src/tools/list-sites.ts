/**
 * List sites tool handler
 */

interface ListSitesArgs {
  type?: string;
  format?: 'summary' | 'full';
}

export async function listSites(args: ListSitesArgs): Promise<string> {
  const { type, format = 'summary' } = args;

  // TODO: Load config and return site list
  const response = {
    sites: [],
    count: 0,
    filter: type,
    format,
  };

  return JSON.stringify(response, null, 2);
}
