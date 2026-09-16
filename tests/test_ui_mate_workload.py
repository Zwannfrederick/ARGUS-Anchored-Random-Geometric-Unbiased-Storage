"""Fast check for the offline UI-Mate grounding scorer; no model or server."""
from pathlib import Path


def test_relative_coordinates_are_scaled_and_validated(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from check_ui_mate_workload import passed

    assert passed("click", {"content": '{"x":780,"y":730}'})
    # Absolute pixels must not silently be treated as normalized coordinates.
    assert not passed("click", {"content": '{"x":400,"y":280}'})
    for value in ('{"x":1000,"y":730}', '{"x":true,"y":730}',
                  '{"x":780.5,"y":730}', '{"x":200,"y":730}', '[]', '{}', 'invalid'):
        assert not passed("click", {"content": value})


def test_reference_action_is_data_not_executable_code(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from check_ui_mate_reference import hits_save

    assert hits_save(["pyautogui.click(400, 280)"])
    for actions in ([], ["pyautogui.click(600, 280)"], ["pyautogui.click(True, 280)"],
                    ["pyautogui.click(400, 280); other()"], ["other.click(400, 280)"],
                    ["pyautogui.click(400, 280)", "other()"], ["pyautogui.click(run(), 280)"]):
        assert not hits_save(actions)


def _run(mode, content, actions, tokens=119):
    return {"mode": mode, "cases": [{"model_grounding": False, "parsed_actions": actions,
            "response": {"usage": {"completion_tokens": tokens},
                         "choices": [{"message": {"content": content, "reasoning_content": "why"}}]}}]}


def test_equivalence_gate_compares_modes_and_never_assumes_a_missing_one(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from check_ui_mate_reference import compare

    action = ["pyautogui.click(383, 239)"]
    stock = _run("stock", "same", action)
    assert compare([stock, _run("argus-cuda", "same", action)])["equivalent"]
    # A missed grounding box is observed, not a failure, when both paths agree.
    assert compare([stock, _run("argus-cuda", "same", action)])["model_grounding"] == {
        "stock": False, "argus-cuda": False}
    for diverged in (_run("argus-cuda", "other", action),
                     _run("argus-cuda", "same", ["pyautogui.click(1, 2)"]),
                     _run("argus-cuda", "same", action, tokens=118)):
        assert not compare([stock, diverged])["equivalent"]
    # One mode alone proves nothing; it must not report an equivalence it never measured.
    single = compare([stock])
    assert not single["compared"] and "equivalent" not in single
