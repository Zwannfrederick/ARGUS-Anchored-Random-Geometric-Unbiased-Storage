export interface StreamedReply {
  content: string;
  reasoning: string;
}

/**
 * Apply complete server-sent events from `buffer` to `reply`; return the unfinished tail.
 * Throws when the gateway or llama-server reports an error event.
 */
export function consumeSse(buffer: string, reply: StreamedReply): string {
  const events = buffer.split('\n\n');
  const rest = events.pop() ?? '';
  for (const event of events) {
    const data = event
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('');
    if (!data || data === '[DONE]') continue;
    const parsed = JSON.parse(data);
    if (parsed.error) {
      throw new Error(typeof parsed.error === 'string' ? parsed.error : parsed.error.message);
    }
    const delta = parsed.choices?.[0]?.delta ?? {};
    reply.content += delta.content ?? '';
    reply.reasoning += delta.reasoning_content ?? '';
  }
  return rest;
}
