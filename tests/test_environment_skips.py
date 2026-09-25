"""Environment skips must happen in setup, or `--no-skips` cannot see them (#30).

`conftest.py` converts a skip into a failure under `--no-skips`, and **only a
setup-phase one**:

    if report.when != "setup":
        return

That asymmetry is deliberate. A skip from a test *body* usually means the case
does not apply — `EA06 has no ORDER BY` — and failing on those would make the
option permanently red and therefore switched off. A skip raised because the
*environment* is wrong is the opposite: it is precisely what CI must not
tolerate, because `python -m etl.download_data` not having run makes most of
this suite skip and pass while testing almost nothing.

The consequence is that an environment skip written in a test body is invisible
to the very option meant to catch it. That is not hypothetical: it was true of
`tests/test_empty_answers.py`'s three regressions and of
`test_ea04_shape_is_not_a_cartesian_product`, all of which built their own
engine and skipped from the body. On a machine without the extension they were
skipped and the suite reported green.

This walks the test modules and fails if an environment skip can reach a test
body. A helper is allowed to skip, provided every caller of it is a fixture —
that is how `tests/test_id_uniqueness.py::embedded_with` is written, and it is
correct.

**Static, deliberately.** Running the suite with the engine removed would prove
the same thing better, and cannot be done from inside it.
"""
from __future__ import annotations

import ast
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent

# What makes a skip an *environment* skip rather than "this case does not
# apply". Matched against the reason string; anything else is assumed to be a
# legitimate body skip, because guessing the other way would fail the suite for
# the wrong reason.
ENVIRONMENT = ("engine unavailable", "download_data", "no TOML reader",
               "catalogue moved")


def _is_fixture(node: ast.FunctionDef) -> bool:
    return any("fixture" in ast.dump(d) for d in node.decorator_list)


def _skips_for_environment(node: ast.FunctionDef) -> list[str]:
    """Reasons this function skips for, if they look environmental."""
    found = []
    for call in ast.walk(node):
        if not (isinstance(call, ast.Call)
                and getattr(call.func, "attr", None) == "skip"):
            continue
        # The reason is the first argument, often an f-string.
        text = ast.dump(call)
        if any(marker in text for marker in ENVIRONMENT):
            found.append(text)
    return found


def test_no_environment_skip_can_reach_a_test_body():
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {n.name: n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef)}
        fixtures = {name for name, n in functions.items() if _is_fixture(n)}
        skippers = {name for name, n in functions.items()
                    if _skips_for_environment(n)}

        for name in sorted(skippers):
            if name in fixtures:
                continue                       # setup phase: what we want
            if name.startswith("test_"):
                offenders.append(f"{path.name}::{name} skips in the test body")
                continue
            # A helper is fine if only fixtures call it.
            body_callers = sorted(
                caller for caller, node in functions.items()
                if caller not in fixtures
                and any(getattr(c.func, "id", None) == name
                        for c in ast.walk(node) if isinstance(c, ast.Call))
            )
            if body_callers:
                offenders.append(
                    f"{path.name}::{name} skips, and is called from "
                    f"{body_callers} which are not fixtures")

    assert not offenders, (
        "environment skips that `--no-skips` cannot convert:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the skip into a fixture. Only setup-phase skips become "
          "failures, so one written in a test body lets a CI job with no "
          "engine — or no `data/` — skip the checks and report green."
    )


def test_this_check_can_actually_fail():
    """Fault injection, since the check above passes by finding nothing.

    A guard whose whole output is "no offenders" is one typo away from being a
    guard over an empty set. This drives the same analysis over a module that
    does exactly what the check forbids.
    """
    source = '''
import pytest

def helper():
    try:
        import samyama
    except Exception as exc:
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

def test_something():
    helper()
'''
    tree = ast.parse(source)
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)}
    assert _skips_for_environment(functions["helper"]), (
        "the reason-matching missed a plain 'engine unavailable' skip, so the "
        "check above is looking for nothing"
    )
    assert not _is_fixture(functions["helper"]), "and it is not a fixture"
