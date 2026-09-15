/**
 * Neo Mobile Companion - End-to-End Live Verification Suite
 * ==========================================================
 * Tests real WebSocket connection, real message loop through E4B Gemma 4,
 * real Wayland desktop screenshot capture, tiered notification push,
 * and replay-resistant high-risk approval flows.
 */

const fs = require('fs');
const path = require('path');

const GATEWAY_URL = 'http://127.0.0.1:8765';
const WS_URL = 'ws://127.0.0.1:8765/ws';
const TOKEN_PATH = path.join(process.env.HOME, '.hermes', 'neo_auth_token.txt');

async function run() {
  console.log('==================================================');
  console.log('NEO MOBILE END-TO-END VERIFICATION');
  console.log('==================================================');

  // 1. Read auth token
  if (!fs.existsSync(TOKEN_PATH)) {
    throw new Error(`Token file not found: ${TOKEN_PATH}`);
  }
  const token = fs.readFileSync(TOKEN_PATH, 'utf-8').trim();
  console.log(`[PASS] Auth Token Loaded: ${token.slice(0, 10)}...`);

  // 2. Test Gateway Health & Dual-Phase Warmup Telemetry
  const healthRes = await fetch(`${GATEWAY_URL}/health`);
  if (!healthRes.ok) throw new Error(`Health check failed: HTTP ${healthRes.status}`);
  const health = await healthRes.json();
  console.log(`[PASS] Gateway Health: ${health.gateway}`);
  console.log(`[PASS] Llama Server Ready: ${health.llama_server.ready}`);
  console.log(`[PASS] Model Warmed: ${health.model_warmed}`);
  if (health.warmup) {
    console.log(`[PASS] Warmup Telemetry: Cold Latency=${health.warmup.cold_action_latency_s}s | Warmed Latency=${health.warmup.warmed_action_latency_s}s | Total=${health.warmup.total_duration_s}s`);
  }
  if (!health.ready || !health.model_warmed) throw new Error('Gateway reported not ready or model not warmed');

  // 3. Create Session
  const sessionRes = await fetch(`${GATEWAY_URL}/api/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ title: 'E2E Companion Session' })
  });
  if (!sessionRes.ok) throw new Error(`Session creation failed: HTTP ${sessionRes.status}`);
  const sessionData = await sessionRes.json();
  const sessionId = sessionData.session.session_id;
  console.log(`[PASS] Session Created: ${sessionId}`);

  // 4. Connect WebSocket with Safe First-Frame Handshake (NO Token in Query String)
  const ws = new WebSocket(WS_URL);
  let authenticated = false;

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('WebSocket auth handshake timed out')), 5000);
    ws.onopen = () => {
      console.log('[PASS] WebSocket Connected (No token in query string), sending auth frame...');
      ws.send(JSON.stringify({ type: 'auth', token: token }));
    };
    ws.onmessage = (msgEvent) => {
      try {
        const data = JSON.parse(msgEvent.data);
        if (data.type === 'auth_ok') {
          authenticated = true;
          clearTimeout(timer);
          console.log(`[PASS] Safe First-Frame Handshake Succeeded (Client: ${data.client_id})`);
          resolve();
        }
      } catch (e) {}
    };
    ws.onerror = (err) => reject(err);
  });

  // Track inbound events and messages
  const inboundMessages = [];
  const inboundEvents = [];
  const approvalEvents = [];

  ws.onmessage = (msgEvent) => {
    try {
      const data = JSON.parse(msgEvent.data);
      if (data.type === 'message') {
        inboundMessages.push(data.message);
      } else if (data.type === 'event') {
        inboundEvents.push(data.event);
      } else if (data.type === 'approval_required') {
        approvalEvents.push(data.approval);
      }
    } catch (e) {}
  };

  // 5. Real Message Travel: Mobile Client -> Neo/Hermes -> E4B Gemma 4 -> Neo -> Mobile
  console.log('\n--- Test: Real Conversational Message Flow ---');
  const userPrompt = 'Neo, tell me about your 128K context orchestrator capabilities in two concise sentences.';
  ws.send(JSON.stringify({
    type: 'send_message',
    session_id: sessionId,
    text: userPrompt,
    client_msg_id: `test_msg_${Date.now()}`
  }));

  // Wait for assistant response
  const startTime = Date.now();
  let assistantReply = null;
  while (Date.now() - startTime < 60000) {
    assistantReply = inboundMessages.find((m) => m.role === 'assistant' && m.session_id === sessionId);
    if (assistantReply && assistantReply.content) break;
    await new Promise((r) => setTimeout(r, 250));
  }

  if (!assistantReply || !assistantReply.content) {
    throw new Error('Did not receive assistant response from E4B Gemma 4 within timeout');
  }
  console.log(`[PASS] Assistant Reply Received (${assistantReply.content.length} chars):`);
  console.log(`"${assistantReply.content.slice(0, 120)}..."`);

  // 6. Real Desktop Screenshot Capture and Download
  console.log('\n--- Test: Real Desktop Screenshot Capture & Transfer ---');
  inboundMessages.length = 0; // reset
  ws.send(JSON.stringify({
    type: 'send_message',
    session_id: sessionId,
    text: 'Please capture a desktop screenshot of the current workspace.',
    client_msg_id: `shot_msg_${Date.now()}`
  }));

  let shotReply = null;
  const shotStart = Date.now();
  while (Date.now() - shotStart < 45000) {
    shotReply = inboundMessages.find((m) => m.screenshot_url);
    if (shotReply) break;
    await new Promise((r) => setTimeout(r, 250));
  }

  if (!shotReply || !shotReply.screenshot_url) {
    throw new Error('Screenshot URL was not returned in message');
  }
  console.log(`[PASS] Screenshot URL returned: ${shotReply.screenshot_url}`);

  // Fetch actual screenshot binary
  const imgRes = await fetch(`${GATEWAY_URL}${shotReply.screenshot_url}`);
  if (!imgRes.ok) throw new Error(`Failed to download screenshot: HTTP ${imgRes.status}`);
  const imgBuffer = await imgRes.arrayBuffer();
  console.log(`[PASS] Screenshot Image Downloaded: ${imgBuffer.byteLength} bytes (Type: ${imgRes.headers.get('content-type')})`);
  if (imgBuffer.byteLength < 50000) {
    throw new Error('Downloaded screenshot is suspiciously small');
  }

  // 7. Notification Push & Urgent Vibration
  console.log('\n--- Test: Tiered Notification & Urgent Vibration Push ---');
  const alertRes = await fetch(`${GATEWAY_URL}/api/events/test_urgent`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!alertRes.ok) throw new Error('Test urgent alert endpoint failed');

  let urgentReceived = null;
  const alertStart = Date.now();
  while (Date.now() - alertStart < 5000) {
    urgentReceived = inboundEvents.find((e) => e.priority === 'urgent');
    if (urgentReceived) break;
    await new Promise((r) => setTimeout(r, 100));
  }

  if (!urgentReceived) {
    throw new Error('Urgent notification was not received over WebSocket');
  }
  console.log(`[PASS] Urgent Event Received: "${urgentReceived.title}"`);
  console.log(`[PASS] Priority: ${urgentReceived.priority.toUpperCase()} (triggers phone vibration)`);

  // 8. HIGH-Risk Approval Flow & Replay Prevention
  console.log('\n--- Test: High-Risk Action Approval & Replay Protection ---');
  ws.send(JSON.stringify({
    type: 'send_message',
    session_id: sessionId,
    text: 'Projedeki geçici tabloları DROP TABLE yap ve diskteki build klasörünü rm -rf ile sil.',
    client_msg_id: `high_risk_${Date.now()}`
  }));

  let pendingApproval = null;
  const apprStart = Date.now();
  while (Date.now() - apprStart < 15000) {
    pendingApproval = approvalEvents.find((a) => a.session_id === sessionId && a.status === 'pending');
    if (pendingApproval) break;
    await new Promise((r) => setTimeout(r, 200));
  }

  if (!pendingApproval) {
    throw new Error('High-risk action did not emit approval_required event');
  }
  const approvalId = pendingApproval.approval_id;
  console.log(`[PASS] Approval Request Emitted: ID=${approvalId} | Risk=${pendingApproval.risk_level}`);
  console.log(`Target Command: "${pendingApproval.command}"`);

  // Tamper Attack Test: Attempt approval with forged/modified action_hash
  const tamperRes = await fetch(`${GATEWAY_URL}/api/approvals/${approvalId}/decide`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ decision: 'approve', action_hash: 'forged_bad_hash_1234567890' })
  });
  console.log(`[PASS] Tamper Attempt HTTP Status: ${tamperRes.status} (Expected: 400 Bad Request)`);
  if (tamperRes.status !== 400) {
    throw new Error(`Tamper protection failed! Expected HTTP 400, got ${tamperRes.status}`);
  }
  const tamperErr = await tamperRes.json();
  console.log(`[PASS] Tamper Protection Error Response: "${tamperErr.error}"`);

  // Send First Legitimate Decision: Approve with verified canonical hash
  const decideRes = await fetch(`${GATEWAY_URL}/api/approvals/${approvalId}/decide`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ decision: 'approve', action_hash: pendingApproval.action_hash })
  });
  if (!decideRes.ok) throw new Error(`Decision submission failed: HTTP ${decideRes.status}`);
  const decideData = await decideRes.json();
  console.log(`[PASS] Legitimate Decision Accepted with Cryptographic Hash: ${JSON.stringify(decideData)}`);

  // Replay Attack Test: Send Second Decision on Same approval_id
  const replayRes = await fetch(`${GATEWAY_URL}/api/approvals/${approvalId}/decide`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ decision: 'approve' })
  });
  console.log(`[PASS] Replay Attempt HTTP Status: ${replayRes.status} (Expected: 409 Conflict)`);
  if (replayRes.status !== 409) {
    throw new Error(`Replay protection failed! Expected HTTP 409, got ${replayRes.status}`);
  }
  const replayErr = await replayRes.json();
  console.log(`[PASS] Replay Protection Error Response: "${replayErr.error}"`);

  // 9. Session Persistence & Continuity
  console.log('\n--- Test: Session Persistence & Continuity ---');
  ws.close();
  const historyRes = await fetch(`${GATEWAY_URL}/api/sessions/${sessionId}/messages`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  const historyData = await historyRes.json();
  console.log(`[PASS] Persistent Messages Recalled: ${historyData.messages.length} messages in database`);
  if (historyData.messages.length < 3) {
    throw new Error('Message count in database does not match conversation history');
  }

  console.log('\n==================================================');
  console.log('ALL NEO MOBILE COMPANION E2E TESTS PASSED (9/9)!');
  console.log('==================================================\n');
}

run().catch((err) => {
  console.error('\n[FAIL] E2E Verification Failed:', err);
  process.exit(1);
});
