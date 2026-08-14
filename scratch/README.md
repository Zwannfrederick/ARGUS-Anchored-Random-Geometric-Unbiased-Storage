# scratch/

Throwaway experiment scripts. **Not part of the package**, not supported, not
run by the test suite, and free to break at any time.

Files here are excluded from pytest collection via `norecursedirs` in
`pyproject.toml`. Several are named `test_*.py` but are manual scripts —
collecting them would inflate the suite's pass count with things nobody
maintains.

Real tests live in `tests/`. If something here is worth keeping, promote it to
a real test there and delete it from this directory.
