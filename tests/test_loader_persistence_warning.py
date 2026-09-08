"""An embedded load is discarded, and the loader has to say so (#4).

`python -m etl.loader` with no `--url` loads ~25K nodes, prints per-label
counts, prints `verified: 25,150 nodes in graph 'default'` and exits 0 -- then
throws all of it away, because `SamyamaClient.embedded()` is in-process and
in-memory. A newcomer following the README and then running
`python -m benchmarks.run_benchmark` (also embedded by default) gets an empty
graph and no explanation.

Everything the loader prints is *true* right up to the moment the process
exits, which is what makes the silence expensive: there is no failure to
notice, only a later command that quietly finds nothing.

These parse the CLI's own text rather than running a load, so they need no
engine and no `data/`. The warning is prose, and prose is what drifts.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOADER = ROOT / "etl" / "loader.py"


def loader_source() -> str:
    return LOADER.read_text(encoding="utf-8")


def warning_block() -> str:
    """The `if not url:` block the loader prints on an embedded run."""
    src = loader_source()
    match = re.search(r"\n    if not url:\n(.*?)\n\nif __name__", src, re.DOTALL)
    assert match, (
        "no `if not url:` block at the end of etl/loader.py. If the warning "
        "moved, update this test with it -- do not delete the test, #4 is "
        "about the warning existing."
    )
    return match.group(1)


def test_an_embedded_load_warns_that_it_was_discarded():
    """The core of #4.

    Asserted on the *presence of the branch*, not on exact wording, so the
    message can be improved without editing this test -- but it cannot vanish.
    """
    block = warning_block()
    assert "err=True" in block, (
        "the embedded-load warning does not go to stderr, so it will be lost in "
        "a pipeline that keeps only stdout"
    )
    lowered = block.lower()
    assert any(word in lowered for word in ("not persisted", "discarded", "gone")), (
        f"the `if not url:` block no longer says the graph is thrown away:\n{block}"
    )


def test_the_warning_names_the_command_that_will_find_nothing():
    """Naming the symptom is what makes it actionable.

    The failure a newcomer actually hits is the *next* command returning
    nothing. A warning that only says "not persisted" leaves them to connect
    the two.
    """
    block = warning_block()
    assert "run_benchmark" in block, (
        "the warning does not name `run_benchmark`, which is the command that "
        "will silently find an empty graph"
    )
    assert "--url" in block, (
        "the warning does not name `--url`, which is the fix"
    )


def test_the_url_help_says_an_embedded_load_is_discarded():
    """`--help` is where someone checks before running, not after."""
    src = loader_source()
    match = re.search(r'"--url".*?help=(.*?)\)\n@click', src, re.DOTALL)
    assert match, "no `--url` option with a help string in etl/loader.py"
    help_text = " ".join(match.group(1).split())
    assert re.search(r"discard|not persist|thrown away", help_text, re.IGNORECASE), (
        f"--url's help does not mention that omitting it discards the load:\n"
        f"  {help_text}"
    )


def test_the_module_docstring_does_not_advertise_the_throwaway_form_first():
    """It used to lead with the form that loses your data.

    The docstring's first usage line is what a reader copies. Listing the
    embedded form first, with no caveat, is how #4 reached a newcomer.
    """
    doc = loader_source().split('"""')[1]
    usage = [ln.strip() for ln in doc.splitlines()
             if ln.strip().startswith("python -m etl.loader")]
    assert usage, "no usage lines in the loader docstring"
    assert "--url" in usage[0], (
        f"the first usage line is `{usage[0]}`, which discards the graph. Lead "
        f"with the `--url` form, or mark the bare one as timing-only."
    )
    assert re.search(r"discard|not persist", doc, re.IGNORECASE), (
        "the module docstring does not say the embedded form is discarded"
    )


@pytest.mark.parametrize("command", ["run_benchmark", "etl.loader"])
def test_the_suggested_recovery_uses_the_same_url_for_both(command):
    """A half-applied fix is the next silent empty graph.

    Pointing only the loader at the server leaves the benchmark embedded, which
    fails exactly as before.
    """
    block = warning_block()
    line = [ln for ln in block.splitlines() if command in ln and "--url" in ln]
    assert line, (
        f"the recovery instructions do not show `{command}` with `--url`; "
        f"applying half of them reproduces #4"
    )
