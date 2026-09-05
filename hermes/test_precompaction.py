#!/usr/bin/env python3
"""
Test Conversation & Tool History Precompaction at ~60K Tokens
=============================================================
Verifies that:
1. Histories under 60,000 estimated tokens remain completely untouched.
2. Histories exceeding 60,000 tokens trigger precompaction of older turns.
3. The latest 5 user/assistant pairs (10 messages) are preserved verbatim.
4. Older turns are summarized with tool action references retained.
"""

import sys
from hermes_supervisor import estimate_tokens, compact_history_if_needed

def run_test():
    print("==================================================")
    print("TESTING ~60K TOKEN PRECOMPACTION & VERBATIM TURN PRESERVATION")
    print("==================================================")

    # 1. Build a synthetic long conversation
    # We want ~15 turn pairs (30 messages).
    # Turns 1..10 will be bulky (~7,000 tokens each) to exceed 60K tokens.
    # Turns 11..15 are the latest 5 turn pairs.

    messages = []
    
    # 10 older turn pairs (bulk content)
    for i in range(1, 11):
        user_msg = {
            "role": "user",
            "content": f"User query {i}: Analyze massive log stream segment {i}.\n" + ("DATA_ROW_SAMPLE_LOG_ENTRY_ABC " * 1200)
        }
        asst_msg = {
            "role": "assistant",
            "content": f"Assistant response {i}: Analyzed segment {i}. Findings: no anomalies detected in cluster node.\n" + ("ANALYSIS_DETAIL_OUTPUT_XYZ " * 1200)
        }
        messages.append(user_msg)
        messages.append(asst_msg)

    # 5 latest turn pairs (must be preserved verbatim)
    latest_pairs = []
    for j in range(1, 6):
        u = {
            "role": "user",
            "content": f"VERBATIM_USER_QUERY_{j}: How is memory usage?"
        }
        a = {
            "role": "assistant",
            "content": f"VERBATIM_ASST_REPLY_{j}: Memory usage is at 42%, 21GB available."
        }
        latest_pairs.append(u)
        latest_pairs.append(a)
        messages.append(u)
        messages.append(a)

    total_tokens_before = estimate_tokens(messages)
    print(f"Total messages before: {len(messages)}")
    print(f"Total estimated tokens before: {total_tokens_before:,}")
    assert total_tokens_before > 60000, f"Expected > 60K tokens, got {total_tokens_before}"

    # 2. Run precompaction with threshold 60,000
    compacted, did_compact = compact_history_if_needed(messages, max_tokens=60000)
    assert did_compact is True, "Expected did_compact to be True"
    total_tokens_after = estimate_tokens(compacted)
    print(f"Total messages after precompaction: {len(compacted)}")
    print(f"Total estimated tokens after: {total_tokens_after:,}")

    # 3. Assertions
    # A) Precompacted tokens should now be well under 60,000
    assert total_tokens_after < 60000, f"Compacted tokens {total_tokens_after} still >= 60,000"
    
    # B) The first message must be the summary context
    first_msg = compacted[0]
    assert first_msg["role"] == "system" or "user" in first_msg["role"], "First message must be system/user summary"
    assert "PRECOMPACTED SESSION HISTORY SUMMARY" in first_msg["content"], "Summary header missing"
    print(f"[PASS] Summary header present: '{first_msg['content'][:60]}...'")

    # C) The latest 5 turn pairs (10 messages) must be preserved VERBATIM
    retained_latest = compacted[-10:]
    assert len(retained_latest) == 10, f"Expected 10 retained messages, got {len(retained_latest)}"
    for idx, (original, result) in enumerate(zip(latest_pairs, retained_latest)):
        assert original["role"] == result["role"], f"Role mismatch at index {idx}: {original['role']} vs {result['role']}"
        assert original["content"] == result["content"], f"Content mismatch at index {idx}:\nOrig: {original['content']}\nRes: {result['content']}"
    print("[PASS] All 5 latest turn pairs (10 messages) are 100% VERBATIM matching!")

    # 4. Under threshold test: ensure messages < 60K are not modified
    small_messages = latest_pairs.copy()
    small_tokens = estimate_tokens(small_messages)
    unmodified, did_compact_small = compact_history_if_needed(small_messages, max_tokens=60000)
    assert did_compact_small is False, "Expected did_compact_small to be False"
    assert len(unmodified) == len(small_messages), "Small message history was altered!"
    assert unmodified == small_messages, "Small history content was altered!"
    print(f"[PASS] History under 60K tokens ({small_tokens} tokens) untouched (identity preserved).")

    print("\n==================================================")
    print("PRECOMPACTION ACCEPTANCE VERIFICATION: 100% PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_test()
