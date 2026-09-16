"""Compare a synthetic click using the pinned upstream UI-Mate prompt/parser.

Requires the reviewed upstream agents/ui_mate_agent.py and demo_workflow.py in
--reference-dir at revision 1cb9e1e44ce856e23b593992b02efbd489943fcb.
Generated action strings are parsed as data, never executed.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys

from check_ui_mate_workload import cases, run


def hits_save(actions):
    if len(actions) != 1:
        return False
    try:
        tree = ast.parse(actions[0])
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return False
        call = tree.body[0].value
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
                or not isinstance(call.func.value, ast.Name) or call.func.value.id != "pyautogui"
                or call.func.attr != "click" or len(call.args) != 2 or call.keywords):
            return False
        if not all(isinstance(arg, ast.Constant) and type(arg.value) is int for arg in call.args):
            return False
        x, y = (arg.value for arg in call.args)
        return 320 <= x <= 480 and 240 <= y <= 320
    except (SyntaxError, TypeError):
        return False


def compare(runs):
    """Stock/ARGUS equivalence gates the run; absolute grounding is only observed.

    A single-mode report cannot be compared, so it reports compared=False instead
    of claiming an equivalence it never measured.
    """
    modes = {run["mode"]: run for run in runs}
    if len(modes) < 2:
        return {"compared": False, "reason": "equivalence needs both modes in one report"}
    stock, argus = (modes[name]["cases"][0] for name in ("stock", "argus-cuda"))

    def message(case):
        return case["response"]["choices"][0]["message"]

    same = {field + "_equal": message(stock).get(field) == message(argus).get(field)
            for field in ("content", "reasoning_content")}
    same["parsed_actions_equal"] = stock["parsed_actions"] == argus["parsed_actions"]
    same["completion_tokens_equal"] = (stock["response"]["usage"]["completion_tokens"]
                                       == argus["response"]["usage"]["completion_tokens"])
    return {"compared": True, **same, "equivalent": all(same.values()),
            "model_grounding": {name: modes[name]["cases"][0]["model_grounding"] for name in modes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "mmproj", "kv-dir", "output", "reference-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--modes", nargs="+", choices=("stock", "argus-cuda"), default=["stock", "argus-cuda"])
    args = parser.parse_args()
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    reference = Path(args.reference_dir).resolve()
    revision = (reference / "revision.txt").read_text().strip()
    if revision != "1cb9e1e44ce856e23b593992b02efbd489943fcb":
        raise ValueError("Use the documented, reviewed reference revision")
    sys.path.insert(0, str(reference))
    from agents.ui_mate_agent import UIMateAgent, parse_response, process_image

    requests, image_hash = cases()
    image_url = requests[1][1]["messages"][0]["content"][0]["image_url"]["url"]
    agent = UIMateAgent(temperature=0, max_tokens=512, enable_thinking=True)
    agent.screenshots.append(process_image(base64.b64decode(image_url.split(",", 1)[1])))
    messages = agent.build_messages("Click the SAVE button.")
    payload = {"messages": messages, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": True}}

    report = {"fixture_sha256": image_hash, "reference_revision": revision,
              "requested_modes": args.modes, "request_timeout_seconds": args.request_timeout,
              "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
              "reference_source": f"https://github.com/Tencent/UI-Mate/tree/{revision}",
              "reference_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (reference / "agents").glob("*.py")},
              "scope": "One synthetic click with upstream message builder, preprocessing and parser; greedy sampling. Actions are not executed. The gate is stock/ARGUS equivalence; model_grounding is a reported UI-Mate quality observation and does not gate. Not a full Neo task or quality benchmark.",
              "runs": []}
    for mode in args.modes:
        # The gate is stock/ARGUS equivalence, so a case passes when the response completed.
        result = run(args, mode, [("reference_click", payload)], lambda name, message: True)
        for case in result["cases"]:
            content = case["response"]["choices"][0]["message"].get("content") or ""
            case["parsed_actions"] = parse_response(content, 512, 384, "relative")[1]
            case["model_grounding"] = hits_save(case["parsed_actions"])
        report["runs"].append(result)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert all("error" not in r and len(r["cases"]) == 1 and r["cases"][0]["passed"]
               for r in report["runs"]), "incomplete reference run; see report"
    report["equivalence"] = compare(report["runs"])
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert report["equivalence"].get("equivalent", not report["equivalence"]["compared"]), \
        "ARGUS changed the reference response; see report"


if __name__ == "__main__":
    main()
