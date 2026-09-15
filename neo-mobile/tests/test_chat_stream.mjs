// Run: node tests/test_chat_stream.mjs (Node strips the TypeScript types natively).
import assert from 'node:assert/strict';
import { consumeSse } from '../src/api/chatStream.ts';

const reply = { content: '', reasoning: '' };
let rest = consumeSse('data: {"choices":[{"delta":{"reasoning_content":"düşün"}}]}\n\ndata: {"choices":[{"del', reply);
assert.equal(reply.reasoning, 'düşün');
assert.equal(reply.content, '');
rest = consumeSse(rest + 'ta":{"content":"Merhaba"}}]}\n\ndata: [DONE]\n\n', reply);
assert.equal(reply.content, 'Merhaba');
assert.equal(rest, '');
assert.throws(() => consumeSse('data: {"error":"Model is busy"}\n\n', reply), /Model is busy/);
assert.throws(() => consumeSse('data: {"error":{"message":"bad"}}\n\n', reply), /bad/);
console.log('chat stream parser: ok');
