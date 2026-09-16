"""Repo-wide pytest configuration.

Exists for one reason: **a skipped test must be able to fail CI** (#2).

13 of the test modules call `pytest.skip` when `data/` is absent, because
`etl.download_data` has to run before anything works. That is right for a
developer who has not downloaded the sources yet, and it is a trap in CI: a
workflow that forgets the download step goes *green* while running almost
nothing.

`pytest --no-skips` turns every *fixture* skip into a failure, so the workflow asserts
that the suite really ran rather than that it merely exited zero.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-skips",
        action="store_true",
        default=False,
        help=("Treat a skipped test as a failure. Used by CI, where a skip "
              "means the environment is wrong rather than that the test is "
              "inapplicable."),
    )


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Convert skips to failures under `--no-skips`.

    `xfail` is deliberately exempt. pytest reports an xfailed test as skipped
    with a `wasxfail` attribute, and this suite has some -- currently the
    `test_demo_recording.py` marks for #30, where the recording predates the
    build. (It also carried six for #56 -- in `test_correctness.py`, four
    parameters across two parametrised sweeps plus the whole-test mark on
    `test_ea04_quantization_unlock_is_not_a_cartesian_product`; and in
    `tests/test_empty_answers.py`, `test_ea01_zero_row_case` -- until the
    embedded engine was pinned to `samyama>=1.7.1` and they all came off.) Those
    are *expected* outcomes that CI should tolerate; a `pytest.skip` for a
    missing `data/` is not. Failing on both would make the option unusable here
    and it would simply be turned off.
    """
    outcome = yield
    report = outcome.get_result()
    if not item.config.getoption("--no-skips"):
        return
    # Only setup-phase skips. A skip raised from a *fixture* means the
    # environment could not be built -- no `data/`, no engine -- which is
    # precisely what CI must not tolerate. A skip raised from the test *body*
    # means the case does not apply, like
    # `tests/test_correctness.py`'s "EA06 has no ORDER BY" for a query with
    # nothing to sort. Failing on the second would make this option permanently
    # red and therefore switched off.
    if report.when != "setup":
        return
    if report.skipped and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = report.longrepr[2]
        report.longrepr = (
            f"Skipped, and --no-skips is set: {reason}\n"
            f"In CI a skip means the environment is wrong -- most often that "
            f"`python -m etl.download_data` has not run, which makes 13 test "
            f"modules skip and the suite pass while testing almost nothing."
        )
