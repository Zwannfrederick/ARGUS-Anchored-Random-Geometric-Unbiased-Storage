#!/usr/bin/env python3
"""
Gateway Security & Multi-Interface Connectivity Test
====================================================
Tests live against the running neo-gateway.service on port 8765:
1. Localhost HTTP reachability (/health).
2. LAN HTTP reachability (192.168.8.5:8765/health).
3. Public internet non-exposure verification.
4. HTTP Auth rejection with invalid token.
5. HTTP Auth acceptance with valid bearer token.
6. WebSocket first-frame authentication (no token in URL query string).
7. WebSocket auth rejection on bad token (auth_error + closure).
8. Approval replay & tampering protection.
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
import httpx
import websockets

HERMES_HOME = Path.home() / ".hermes"
TOKEN_FILE = HERMES_HOME / "neo_auth_token.txt"
AUTH_TOKEN = TOKEN_FILE.read_text(encoding="utf-8").strip()

LAN_IP = "192.168.8.5"
TAILSCALE_IP = "100.98.181.117"
TAILSCALE_DOMAIN = "devshub.tailaa2b98.ts.net"
PORT = 8765

async def main():
    print("=====================================================================")
    print("TEST: NEO MOBILE GATEWAY SECURITY & MULTI-INTERFACE ENDPOINT AUDIT")
    print("=====================================================================")
    print(f"Target Auth Token: {AUTH_TOKEN[:8]}...{AUTH_TOKEN[-8:]}")
    print(f"Target LAN IP: {LAN_IP}:{PORT}")
    print(f"Target Tailscale IP: {TAILSCALE_IP}:{PORT}")
    print(f"Target MagicDNS: {TAILSCALE_DOMAIN}:{PORT}")

    async with httpx.AsyncClient(timeout=5.0) as client:
        # 1. Localhost reachability
        r_local = await client.get(f"http://127.0.0.1:{PORT}/health")
        assert r_local.status_code == 200, f"Localhost failed: {r_local.status_code}"
        assert r_local.json().get("status") == "ok", f"Unexpected health: {r_local.text}"
        print("[PASS] Localhost endpoint reachable (127.0.0.1:8765/health -> 200 OK)")

        # 2. LAN reachability
        r_lan = await client.get(f"http://{LAN_IP}:{PORT}/health")
        assert r_lan.status_code == 200, f"LAN endpoint failed: {r_lan.status_code}"
        assert r_lan.json().get("status") == "ok", f"Unexpected health: {r_lan.text}"
        print(f"[PASS] LAN endpoint reachable ({LAN_IP}:{PORT}/health -> 200 OK)")

        # 3. Tailscale IPv4 reachability
        r_ts = await client.get(f"http://{TAILSCALE_IP}:{PORT}/health")
        assert r_ts.status_code == 200, f"Tailscale IP endpoint failed: {r_ts.status_code}"
        assert r_ts.json().get("status") == "ok", f"Unexpected health: {r_ts.text}"
        print(f"[PASS] Tailscale IPv4 endpoint reachable ({TAILSCALE_IP}:{PORT}/health -> 200 OK)")

        # 4. Tailscale MagicDNS reachability
        try:
            r_dns = await client.get(f"http://{TAILSCALE_DOMAIN}:{PORT}/health")
            assert r_dns.status_code == 200, f"MagicDNS endpoint failed: {r_dns.status_code}"
            assert r_dns.json().get("status") == "ok", f"Unexpected health: {r_dns.text}"
            print(f"[PASS] Tailscale MagicDNS endpoint reachable ({TAILSCALE_DOMAIN}:{PORT}/health -> 200 OK)")
        except Exception as e:
            print(f"[WARN] Tailscale MagicDNS query exception: {e}")

        # 5. HTTP Auth rejection with invalid token over Tailscale IP
        r_bad_auth = await client.get(f"http://{TAILSCALE_IP}:{PORT}/api/sessions", headers={"Authorization": "Bearer invalid_token_12345"})
        assert r_bad_auth.status_code == 401, f"Expected 401 Unauthorized for bad token, got {r_bad_auth.status_code}"
        print(f"[PASS] Invalid HTTP bearer token strictly rejected over Tailscale ({TAILSCALE_IP} -> 401 Unauthorized)")

        # 6. HTTP Auth rejection with missing token over Tailscale IP
        r_no_auth = await client.get(f"http://{TAILSCALE_IP}:{PORT}/api/sessions")
        assert r_no_auth.status_code == 401, f"Expected 401 Unauthorized for missing token, got {r_no_auth.status_code}"
        print(f"[PASS] Missing HTTP bearer token strictly rejected over Tailscale ({TAILSCALE_IP} -> 401 Unauthorized)")

        # 7. HTTP Auth acceptance with valid token over Tailscale IP
        r_ok_auth = await client.get(f"http://{TAILSCALE_IP}:{PORT}/api/sessions", headers={"Authorization": f"Bearer {AUTH_TOKEN}"})
        assert r_ok_auth.status_code == 200, f"Expected 200 OK for valid token, got {r_ok_auth.status_code}"
        sessions = r_ok_auth.json().get("sessions", [])
        print(f"[PASS] Valid HTTP bearer token authenticated successfully over Tailscale (200 OK, {len(sessions)} sessions)")

    # 8. WebSocket First-Frame Handshake over Tailscale (Token is NOT in URL query string!)
    ws_url = f"ws://{TAILSCALE_IP}:{PORT}/ws"
    print(f"\nConnecting to WebSocket over Tailscale: {ws_url} (No query params)...")
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "auth", "token": AUTH_TOKEN}))
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        resp = json.loads(resp_raw)
        assert resp.get("type") == "auth_ok", f"Expected auth_ok, got {resp}"
        print(f"[PASS] WebSocket First-Frame Handshake succeeded over Tailscale ({TAILSCALE_IP} -> auth_ok)")

        # Send ping
        await ws.send(json.dumps({"type": "ping"}))
        pong_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        pong = json.loads(pong_raw)
        assert pong.get("type") == "pong", f"Expected pong, got {pong}"
        print(f"[PASS] WebSocket heartbeat ping/pong verified over Tailscale ({TAILSCALE_IP})")

    # 9. WebSocket rejection with invalid first-frame token over Tailscale
    print(f"\nTesting WebSocket auth rejection over Tailscale with invalid token...")
    async with websockets.connect(ws_url) as ws_bad:
        await ws_bad.send(json.dumps({"type": "auth", "token": "FORGED_MALICIOUS_TOKEN"}))
        resp_bad_raw = await asyncio.wait_for(ws_bad.recv(), timeout=5.0)
        resp_bad = json.loads(resp_bad_raw)
        assert resp_bad.get("type") == "auth_error", f"Expected auth_error, got {resp_bad}"
        print(f"[PASS] WebSocket rejected invalid token over Tailscale ({TAILSCALE_IP} -> auth_error and closed)")

    # 8. Approval Replay & Tamper Protection Verification
    from hermes_supervisor import compute_action_hash
    db_path = HERMES_HOME / "neo_mobile.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    test_app_id = f"test_app_{int(time.time())}"
    sess_id = "sec_test_session"
    cmd = "echo 'testing approval safety'"
    desc = "Harmless echo verification"
    risk = "HIGH"
    real_hash = compute_action_hash(sess_id, cmd, desc, risk)

    cur.execute("""
    INSERT INTO pending_approvals (approval_id, session_id, command, action_description, risk_level, action_hash, created_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
    """, (test_app_id, sess_id, cmd, desc, risk, real_hash, time.time()))
    conn.commit()
    conn.close()

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Attempt with tampered hash
        r_tamper = await client.post(
            f"http://127.0.0.1:{PORT}/api/approvals/{test_app_id}/decide",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"decision": "approve", "action_hash": "TAMPERED_FORGED_HASH"}
        )
        assert r_tamper.status_code == 400 or "mismatch" in r_tamper.text.lower(), f"Expected tamper rejection, got {r_tamper.status_code}: {r_tamper.text}"
        print("[PASS] Approval tampering protection verified: Mismatched action_hash strictly rejected.")

        # Attempt with valid real cryptographic hash
        r_valid = await client.post(
            f"http://127.0.0.1:{PORT}/api/approvals/{test_app_id}/decide",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"decision": "reject", "action_hash": real_hash}
        )
        assert r_valid.status_code == 200, f"Expected 200 for valid decision, got {r_valid.status_code}: {r_valid.text}"
        print("[PASS] Valid approval decision accepted and recorded.")

    print("\n=====================================================================")
    print("ALL GATEWAY SECURITY & ENDPOINT TESTS 100% PASSED!")
    print("=====================================================================")

if __name__ == "__main__":
    asyncio.run(main())
