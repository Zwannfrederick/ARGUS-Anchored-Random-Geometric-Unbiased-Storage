"""Opt-in UI-Mate stock/ARGUS CUDA comparison; tool calls are never executed.

Run with --server, --model, --mmproj, --kv-dir and --output paths.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from bench_llama_paged_context import http_json


def cases():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (512, 384), "#eeeeee")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    draw.text((24, 32), "Settings", fill="black", font=font)
    for box, label in [((32, 240, 192, 320), "CANCEL"), ((320, 240, 480, 320), "SAVE")]:
        draw.rectangle(box, fill="white", outline="black", width=3)
        draw.text((box[0] + 20, 265), label, fill="black", font=font)
    data = io.BytesIO()
    image.save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    visual = [{"type": "image_url", "image_url": {"url": url}}]
    tool = {"type": "function", "function": {"name": "ui_click", "description": "Click the named button.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
    requests = [
        ("turkish", {"messages": [{"role": "user", "content": "Türkiye'nin başkenti hangi şehirdir? Yalnızca şehir adını yaz."}]}),
        ("grounding", {"messages": [{"role": "user", "content": visual + [{"type": "text", "text": "List the labels of the two buttons in this image, left to right. Output labels only."}]}]}),
        ("click", {"messages": [{"role": "user", "content": visual + [{"type": "text", "text": "Locate the center of the SAVE button. Use normalized integer coordinates from 0 to 999 for both axes, with (0,0) at top left and (999,999) at bottom right. Return only JSON with x and y."}]}],
                   "response_format": {"type": "json_schema", "json_schema": {"name": "click_point", "strict": True,
                       "schema": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                                  "required": ["x", "y"], "additionalProperties": False}}}}),
        ("tool_call", {"messages": [{"role": "user", "content": visual + [{"type": "text", "text": "Click the SAVE button using ui_click with its text label."}]}], "tools": [tool], "tool_choice": "required"}),
    ]
    return requests, hashlib.sha256(data.getvalue()).hexdigest()


def passed(name, message):
    content = message.get("content") or ""
    if name == "turkish":
        return "ankara" in content.casefold()
    if name == "grounding":
        upper = content.upper()
        return "CANCEL" in upper and "SAVE" in upper and upper.index("CANCEL") < upper.index("SAVE")
    if name == "click":
        try:
            point = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            if not all(type(point[k]) is int and 0 <= point[k] <= 999 for k in ("x", "y")):
                return False
            # UI-Mate's reference adapter uses x*width/999 and y*height/999.
            return 320 <= int(point["x"] * 512 / 999) <= 480 and 240 <= int(point["y"] * 384 / 999) <= 320
        except (ValueError, KeyError, TypeError):
            return False
    try:
        calls = message.get("tool_calls", [])
        return len(calls) == 1 and calls[0]["function"]["name"] == "ui_click" and json.loads(calls[0]["function"]["arguments"])["text"].casefold() == "save"
    except (ValueError, KeyError, TypeError):
        return False


def run(args, mode, requests, score=passed):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARGUS_KV_")}
    command = [args.server, "-m", args.model, "--mmproj", args.mmproj, "--no-mmproj-offload",
               "--image-min-tokens", "1024",
               "-c", "4096", "-np", "1", "-ngl", "8", "-nkvo", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
               "-b", "512", "-ub", "32", "-t", "6", "--no-webui", "--host", "127.0.0.1", "--port", str(port)]
    result = {"mode": mode, "command": command, "cases": []}
    with tempfile.TemporaryDirectory(dir=args.kv_dir) as storage:
        stats = Path(storage) / "stats.json"
        if mode == "argus-cuda":
            env.update(ARGUS_KV_DIR=storage, ARGUS_KV_MAX_BYTES=str(1 << 30), ARGUS_KV_RESIDENT_BYTES=str(8 << 20),
                       ARGUS_KV_STAGING_BYTES=str(4 << 20), ARGUS_KV_GPU_BYTES=str(4 << 20),
                       ARGUS_KV_PINNED_BYTES=str(4 << 20), ARGUS_KV_STATS_PATH=str(stats))
        result["argus_environment"] = {k: v for k, v in env.items() if k.startswith("ARGUS_KV_")}
        log_path = Path(args.output).with_suffix(f".{mode}.log")
        with log_path.open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during startup")
                    try:
                        if http_json(base + "/health", timeout=2).get("status") == "ok":
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server startup")
                    time.sleep(0.5)
                for name, payload in requests:
                    started = time.monotonic()
                    response = http_json(base + "/v1/chat/completions", {
                        "temperature": 0, "seed": 42, "max_tokens": 128,
                        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}, **payload},
                        timeout=getattr(args, "request_timeout", 600))
                    message = response["choices"][0]["message"]
                    complete = response["choices"][0].get("finish_reason") in ("stop", "tool_calls")
                    result["cases"].append({"name": name, "passed": complete and score(name, message), "response": response,
                                            "seconds": time.monotonic() - started})
                    print(mode, name, result["cases"][-1]["passed"], flush=True)
                if stats.exists():
                    result["stats"] = json.loads(stats.read_text())
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                result["error"] = f"{type(error).__name__}: {error}"
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                result["exit_code"] = process.returncode
                if stats.exists():
                    # Stats are published during attention, not by the destructor.
                    result["last_published_stats_after_exit"] = json.loads(stats.read_text())
        result["log"] = str(log_path)
        result["teardown_log"] = [line for line in log_path.read_text().splitlines() if "ARGUS_DISK free" in line]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "mmproj", "kv-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    requests, image_hash = cases()
    report = {"fixture_sha256": image_hash, "scope": "Four fixed prompts; click uses 0-999 relative coordinates and server JSON schema. Simulated clicks, no tools executed. Not a quality benchmark or policy performance claim.",
              "coordinate_contract_source": "https://github.com/Tencent/UI-Mate/blob/main/agents/ui_mate_agent.py",
              "llama_revision": "54315813269112dd0baed7112ec87ad93a8218ca", "runs": []}
    for mode in ("stock", "argus-cuda"):
        report["runs"].append(run(args, mode, requests))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert all("error" not in r and len(r["cases"]) == 4 for r in report["runs"]), "incomplete comparison; see report"
    report["comparison"] = [{"case": stock["name"], "stock_passed": stock["passed"],
        "argus_passed": argus["passed"],
        "content_equal": stock["response"]["choices"][0]["message"].get("content") == argus["response"]["choices"][0]["message"].get("content")}
        for stock, argus in zip(report["runs"][0]["cases"], report["runs"][1]["cases"], strict=True)]
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert all(c["stock_passed"] and c["argus_passed"] for c in report["comparison"]), "semantic check failed; see report"
    stats = report["runs"][1]["stats"]
    assert stats["cuda_attention_calls"] > 0
    assert all(stats["peak_" + tier + "_bytes"] <= 4 << 20 for tier in ("gpu", "pinned", "staging"))


if __name__ == "__main__":
    main()
