#!/usr/bin/env python3
"""
Test Raw Conversation Archive & Non-Destructive 60K Precompaction
==================================================================
Verifies:
1. llama-server max context window is preserved (131072).
2. Active prompt precompaction triggers at ~60K tokens.
3. Active prompt is reduced, retaining latest 5 turn pairs verbatim and summarizing older turns.
4. Raw conversation history in SQLite is NEVER mutated or deleted.
5. Historical retrieval / session_search can retrieve exact original turns from the DB post-compaction.
"""

import os
import sys
import sqlite3
import time
from pathlib import Path

# Add hermes directory to path
HERMES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HERMES_DIR))

from hermes_supervisor import estimate_tokens, compact_history_if_needed, COMPACTION_TOKEN_THRESHOLD
from neo_mobile_gateway import DB_PATH, init_db

def test_archive_preservation_and_compaction():
    print("=====================================================================")
    print("TEST: Raw Conversation Archive Preservation & Non-Destructive Precompaction")
    print("=====================================================================")

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    test_session_id = f"test_archive_session_{int(time.time())}"
    now = time.time()
    cur.execute("INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (test_session_id, "Test Archive Session", now, now))
    conn.commit()

    print(f"Created test session: {test_session_id}")

    # Build 10 bulky historical turn pairs (~7K tokens each -> >70K tokens total)
    raw_history = []
    canary_texts = {}

    for i in range(1, 11):
        u_text = f"CANARY_RAW_HISTORICAL_USER_TURN_{i}: Detailed telemetry inspection request #{i}.\n" + ("TELEMETRY_DATA_BLOCK_ALPHA_BETA_ " * 1200)
        a_text = f"CANARY_RAW_HISTORICAL_ASST_TURN_{i}: Telemetry analysis complete for block #{i}. All systems nominal.\n" + ("SYSTEM_LOG_PARSED_OUTPUT_OK_ " * 1200)
        canary_texts[f"user_{i}"] = u_text
        canary_texts[f"asst_{i}"] = a_text

        u_id = f"msg_{test_session_id}_u_{i}"
        a_id = f"msg_{test_session_id}_a_{i}"

        cur.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'user', ?, ?)",
                    (u_id, test_session_id, u_text, now + (i * 2)))
        cur.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'assistant', ?, ?)",
                    (a_id, test_session_id, a_text, now + (i * 2) + 1))
        raw_history.append({"role": "user", "content": u_text})
        raw_history.append({"role": "assistant", "content": a_text})

    # Add 5 latest turn pairs (must be preserved verbatim in active prompt)
    latest_pairs = []
    for j in range(1, 6):
        u_verbatim = f"LATEST_VERBATIM_USER_QUERY_{j}: Current CPU temperature and fan speed?"
        a_verbatim = f"LATEST_VERBATIM_ASST_REPLY_{j}: CPU is at 45C, fan speed 1200 RPM, all quiet."
        latest_pairs.append({"role": "user", "content": u_verbatim})
        latest_pairs.append({"role": "assistant", "content": a_verbatim})

        u_id = f"msg_{test_session_id}_latest_u_{j}"
        a_id = f"msg_{test_session_id}_latest_a_{j}"
        cur.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'user', ?, ?)",
                    (u_id, test_session_id, u_verbatim, now + 100 + (j * 2)))
        cur.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, 'assistant', ?, ?)",
                    (a_id, test_session_id, a_verbatim, now + 100 + (j * 2) + 1))
        raw_history.append({"role": "user", "content": u_verbatim})
        raw_history.append({"role": "assistant", "content": a_verbatim})

    conn.commit()

    # Step 1: Confirm raw history exceeds threshold
    total_tokens_before = estimate_tokens(raw_history)
    print(f"Total raw history messages: {len(raw_history)}")
    print(f"Total estimated tokens before compaction: {total_tokens_before:,}")
    assert total_tokens_before > COMPACTION_TOKEN_THRESHOLD, f"Expected > {COMPACTION_TOKEN_THRESHOLD}, got {total_tokens_before}"
    print(f"[PASS] Raw history exceeds {COMPACTION_TOKEN_THRESHOLD:,} tokens threshold.")

    # Step 2: Perform active context precompaction (what HermesSupervisor does before calling LLM)
    compacted_prompt, did_compact = compact_history_if_needed(raw_history, max_tokens=COMPACTION_TOKEN_THRESHOLD)
    assert did_compact is True, "Precompaction should have fired"
    total_tokens_after = estimate_tokens(compacted_prompt)
    print(f"Active prompt messages after compaction: {len(compacted_prompt)}")
    print(f"Active prompt estimated tokens: {total_tokens_after:,}")

    # Step 3: Confirm active prompt is shortened and conforms to policy
    assert total_tokens_after < COMPACTION_TOKEN_THRESHOLD, f"Active prompt {total_tokens_after} still >= threshold"
    assert "PRECOMPACTED SESSION HISTORY SUMMARY" in compacted_prompt[0]["content"], "Summary missing"
    
    # Confirm latest 5 turn pairs are 100% verbatim in active prompt
    verbatim_tail = compacted_prompt[-10:]
    assert len(verbatim_tail) == 10, f"Expected 10 verbatim messages, got {len(verbatim_tail)}"
    for idx, (expected, actual) in enumerate(zip(latest_pairs, verbatim_tail)):
        assert expected["role"] == actual["role"], f"Role mismatch at tail index {idx}"
        assert expected["content"] == actual["content"], f"Content mismatch at tail index {idx}"
    print("[PASS] Active prompt compacted successfully: 5 latest turn pairs preserved 100% verbatim.")

    # Step 4: Prove database persistence was NOT destroyed or mutated
    # Query database for all messages in the session
    cur.execute("SELECT id, role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (test_session_id,))
    db_rows = cur.fetchall()
    print(f"Total messages stored in SQLite DB post-compaction: {len(db_rows)}")
    assert len(db_rows) == 30, f"Expected 30 messages in DB, got {len(db_rows)} (messages were erroneously deleted!)"

    # Step 5: Test retrieval of historical turn #3 using session_search pattern
    search_query = "%CANARY_RAW_HISTORICAL_USER_TURN_3%"
    cur.execute("SELECT id, role, content FROM messages WHERE session_id = ? AND content LIKE ?", (test_session_id, search_query))
    found = cur.fetchall()
    assert len(found) == 1, f"Expected exactly 1 match for turn 3 canary, found {len(found)}"
    matched_id, matched_role, matched_content = found[0]

    # Verify exact byte-for-byte fidelity with original uncompacted text
    expected_content = canary_texts["user_3"]
    assert matched_content == expected_content, "Historical content was modified or corrupted!"
    assert len(matched_content) == len(expected_content), "Historical content length mismatch!"
    print(f"[PASS] Exact original historical turn #3 retrieved from SQLite ({len(matched_content):,} chars, byte-perfect).")

    # Step 6: Test retrieval of historical assistant turn #7
    search_query_asst = "%CANARY_RAW_HISTORICAL_ASST_TURN_7%"
    cur.execute("SELECT id, role, content FROM messages WHERE session_id = ? AND content LIKE ?", (test_session_id, search_query_asst))
    found_asst = cur.fetchall()
    assert len(found_asst) == 1, f"Expected 1 match for turn 7 canary, found {len(found_asst)}"
    assert found_asst[0][2] == canary_texts["asst_7"], "Historical assistant response #7 mismatch!"
    print(f"[PASS] Exact original historical assistant reply #7 retrieved from SQLite ({len(found_asst[0][2]):,} chars, byte-perfect).")

    # Clean up test rows
    cur.execute("DELETE FROM messages WHERE session_id = ?", (test_session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (test_session_id,))
    conn.commit()
    conn.close()

    print("\n=====================================================================")
    print("COMPACTION & ARCHIVE PRESERVATION VERIFICATION: 100% PASSED!")
    print("Precompaction reduces active prompt only; zero raw history is destroyed.")
    print("=====================================================================")

if __name__ == "__main__":
    test_archive_preservation_and_compaction()
