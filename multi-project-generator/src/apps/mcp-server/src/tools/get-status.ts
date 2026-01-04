/**
 * Get status tool handler
 */

interface GetStatusArgs {
  detailed?: boolean;
}

export async function getStatus(args: GetStatusArgs): Promise<string> {
  const { detailed = false } = args;

  // TODO: Get actual generation status
  const response = {
    status: 'idle',
    sites: {
      total: 0,
      pending: 0,
      completed: 0,
      failed: 0,
    },
    detailed,
  };

  return JSON.stringify(response, null, 2);
}
