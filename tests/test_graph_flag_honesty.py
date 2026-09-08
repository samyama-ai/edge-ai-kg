"""`--graph` is accepted and ignored, and the help text has to say so (#82).

`docs/engine-notes.md:222` records that the tenant argument is ignored on the
OSS HTTP path -- everything lands in `default` whatever is passed. Both CLIs
still advertised it as "Target graph / tenant." with nothing to warn a reader,
so `--help` promised a boundary the engine does not enforce.

The failure mode is quiet and bad: `--graph tenant-a` produces no error, no
warning, and the caller's data sits in `default` alongside everyone else's.

`tests/test_cli_defaults.py` cannot catch this, and the reason is worth stating.
It pins that every entry point *defaults* to `"default"` -- which is exactly why
the disagreement is invisible. The flag being ignored means every code path
behaves identically whatever is passed, so a defaults test passes either way.

This asserts the sentence instead, because the sentence is the only thing
standing between a reader and a wrong assumption, and a sentence is what drifts.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every CLI that exposes the option.
CLIS = ("etl/loader.py", "benchmarks/run_benchmark.py")

# Any of these in the help text counts as disclosing it. A list rather than an
# exact string so the wording can be improved without editing this test.
DISCLOSURES = ("ignored", "not enforced", "no effect", "has no effect")


def graph_option_help(relative_path: str) -> str:
    """The `help=` text of the `--graph` option, across line continuations."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    match = re.search(
        r'@click\.option\(\s*"--graph".*?help="(.*?)"\s*\)',
        source, re.DOTALL)
    assert match, (
        f"no `--graph` option with a help string found in {relative_path}. If "
        f"the option was removed, that also resolves #82 -- delete this "
        f"parameter from CLIS."
    )
    return " ".join(match.group(1).split())


@pytest.mark.parametrize("cli", CLIS)
def test_the_graph_option_says_it_is_ignored(cli):
    """The disclosure, at the point a user reads it.

    Not in `docs/engine-notes.md` -- it is already there, note 7, and that did
    not help anyone typing `--help`.
    """
    text = graph_option_help(cli)
    assert any(word in text.lower() for word in DISCLOSURES), (
        f"{cli}'s --graph help does not disclose that the argument is ignored:\n"
        f"  {text!r}\n"
        f"On the OSS build everything lands in `default` regardless (engine "
        f"note 7), so this text promises tenant isolation that does not exist."
    )


@pytest.mark.parametrize("cli", CLIS)
def test_the_disclosure_points_at_the_engine_note(cli):
    """A reader who does not believe it should be able to check.

    Cheap, and it is what turns the warning from an assertion into something
    verifiable -- which is the same standard the rest of the docs are held to.
    """
    text = graph_option_help(cli)
    assert re.search(r"note\s*7", text, re.IGNORECASE), (
        f"{cli}'s --graph help says the argument is ignored but does not cite "
        f"engine note 7, so a reader cannot check the claim:\n  {text!r}"
    )
