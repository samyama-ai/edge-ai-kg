"""`docs/why-this-engine.md` must keep the contract it sets itself (#43, #51, #52).

That page opens by promising every claim is one of three kinds, each labelled:

> - **Measured** -- a command in this repo produces the number.
> - **Quoted** -- taken verbatim from the other project's own licence file, linked.
> - **Unmeasured** -- believed, not tested. Named as such, with the open issue.

A competitor comparison is the easiest document in a repo to drift into
marketing, and the only thing standing between this one and that is whether
those labels stay true. Prose cannot enforce itself, so:

**Every `Measured` row names a command, and the command exists.** Before this
test, four rows cited an issue number instead -- `#18`, `#45`, `#48`, and one
that said outright "no command on this branch". An issue number is a place to
read about a measurement, not a way to reproduce it, which is exactly the
distinction #43 asks for.

**Every `Quoted` row carries a link**, because #51's requirement is that no
claim about another project's terms is made that is not quoted from that
project's own documents.

**Every competitor named says both what it does well and when to choose it**,
which is #52's requirement and the one a reader checks first for honesty.

The tests read the document rather than a copy of its claims. A row edited in
the page is a row this starts checking; a row deleted is one it stops asking
about. That is deliberate -- a hand-maintained list here would be a second
thing to keep in sync, and the failure mode of these tests would become "the
list is stale" rather than "the page is wrong".
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "why-this-engine.md"

# `python -m benchmarks.x --flag`, `pytest tests/y.py`. The flags are captured
# too: the module path was the only thing checked, so `--cold-start` could be
# deleted from `benchmarks/ingest.py` and the row citing it stayed green -- the
# file still existed. A command that does not accept the flag the page prints is
# not a reproduction.
COMMAND = re.compile(r"`(python -m [\w.]+(?:\s+--[\w-]+)*|pytest [\w/.:]+)")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")


def claims_table() -> list[tuple[str, str, str]]:
    """The `| Claim | Status | How |` rows, as `(claim, status, how)`."""
    text = DOC.read_text(encoding="utf-8")
    block = text.split("| Claim | Status | How |", 1)
    assert len(block) == 2, (
        "the claims table header is gone from docs/why-this-engine.md. If it "
        "moved or was renamed, fix this parser -- do not delete the check, "
        "which is the only thing keeping the labels honest."
    )
    rows = []
    for line in block[1].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue                        # the |---|---|---| separator
        # Fail rather than skip. Skipping let a row with an escaped pipe parse
        # as four cells and slip past every check below -- a Measured row could
        # name a module that does not exist and all seven tests stayed green.
        assert len(cells) == 3, (
            f"claims-table row does not have three cells: {line!r}. If it "
            f"contains an escaped pipe, this parser needs to handle it -- "
            f"skipping the row would exempt it from every check here."
        )
        rows.append(tuple(cells))
    # Counted on rows with a claim. Continuation rows -- `| | | detail |`,
    # carrying a second line of `How` for the row above -- were counted toward
    # this floor, so a table of six real claims and two continuations passed a
    # check meant to require eight claims.
    claims = [r for r in rows if r[0]]
    assert len(claims) >= 8, (
        f"only {len(claims)} claim rows parsed ({len(rows)} table rows "
        f"including continuations); the table shape changed"
    )
    return rows


def click_options(module: str) -> set[str] | None:
    """The long options the module's click command declares, or `None`.

    Asked of click rather than grepped for the literal string: a flag mentioned
    in a docstring, a help text or a `raise SystemExit` message would satisfy a
    grep while `main()` rejected it, which is the failure this is here to catch.
    """
    import importlib

    import click

    mod = importlib.import_module(module)
    # `main`, by name. `next(v for v in vars(mod).values() if isinstance(...))`
    # depends on definition order and would silently check an *imported*
    # command if a module ever imported one from elsewhere -- reporting the
    # flags of the wrong program.
    command = getattr(mod, "main", None)
    if not isinstance(command, click.Command):
        return None
    declared = {opt for param in command.params for opt in param.opts
                if opt.startswith("--")}
    # `--help` is added by click to every command and is not in `params`, so a
    # row citing it is reproducible even though nothing declares it.
    return declared | {"--help"}


def test_every_measured_claim_names_a_command_that_exists():
    """#43: "every claim in it is reproducible by a command in the repo"."""
    problems = []
    for claim, status, how in claims_table():
        if "Measured" not in status or "Unmeasured" in status:
            continue
        if not claim:                      # continuation row carrying detail
            continue
        found = COMMAND.findall(how)
        if not found:
            # No exemption. This used to pass any row whose `How` contained the
            # substring "no command" -- so a row could be labelled **Measured**,
            # which the page defines as "a command in this repo produces the
            # number", and then say in the same cell that no command exists.
            # Any future row could take that exit by typing three words. A row
            # with no command is **Quoted**, which is a label the page offers.
            problems.append(f"{claim!r}: labelled Measured, which this page "
                            f"defines as 'a command in this repo produces the "
                            f"number', and names none -- How = {how!r}. If "
                            f"there is no command, the label is Quoted.")
            continue
        for command in found:
            if command.startswith("python -m "):
                module, *flags = command[len("python -m "):].split()
                path = ROOT / (module.replace(".", "/") + ".py")
                if not path.exists():
                    problems.append(f"{claim!r}: `{command}` -- "
                                    f"{path.relative_to(ROOT)} does not exist")
                    continue
                declared = click_options(module)
                for flag in flags:
                    # `declared is None` means the module has no click command
                    # to ask -- reported, not skipped, because a silent skip is
                    # how the module-only check stayed green for so long.
                    if declared is None:
                        problems.append(f"{claim!r}: `{command}` -- {module} "
                                        f"exposes no click command, so `{flag}` "
                                        f"cannot be checked")
                        break
                    if flag not in declared:
                        problems.append(
                            f"{claim!r}: `{command}` -- {module} does not accept "
                            f"`{flag}`. It accepts {sorted(declared)}.")
            else:
                target = command.split(" ", 1)[1].split("::")[0]
                if not (ROOT / target).exists():
                    problems.append(f"{claim!r}: `{command}` -- {target} does not exist")
    assert not problems, (
        "docs/why-this-engine.md claims to be reproducible and is not:\n  "
        + "\n  ".join(problems)
    )


def test_every_doc_link_in_the_claims_table_resolves():
    """A pointer to a page that does not exist is not a reproduction either.

    The snapshot row named its command and then sent the reader to
    `neo4j-comparison.md` "for the full invocation". `docs/` had no such file --
    it is created on another branch. So the one row admitting it needs an
    invocation this repo does not carry pointed at a page this repo does not
    carry, on a page whose whole argument is that a pointer to nowhere is not
    evidence. The command check walked straight past it: `COMMAND` matches only
    backticked `python -m` and `pytest`, and nothing looked at link targets.

    External links are not fetched -- that would make the suite depend on the
    network. Only repo-relative targets are resolved.
    """
    problems = []
    for claim, _status, how in claims_table():
        for target in re.findall(r"\]\(([^)]+)\)", how):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (DOC.parent / target.split("#")[0]).resolve()
            if not path.exists():
                problems.append(f"{claim or '(continuation row)'}: "
                                f"links to {target!r}, which does not exist")
    assert not problems, (
        "docs/why-this-engine.md links to files that are not here:\n  "
        + "\n  ".join(problems)
    )


def test_no_claim_is_unlabelled():
    """The three labels are the contract; a fourth kind is a claim in disguise."""
    # The three the page's own opening offers, and no more. "Available" was a
    # fourth, and it had a lowercase `measured` beside it -- so
    # `test_every_measured_claim_names_a_command_that_exists` skipped that row
    # and nothing checked the command it named.
    allowed = ("Measured", "Unmeasured", "Quoted")
    stray = [f"{claim!r} -> {status!r}" for claim, status, _ in claims_table()
             if claim and not any(word in status for word in allowed)]
    assert not stray, (
        f"claims carrying no recognised label: {stray}. The page's own opening "
        f"says every claim is Measured, Quoted or Unmeasured."
    )


def test_every_quoted_licence_claim_links_to_the_source():
    """#51: nothing said about another project's terms that is not quoted from it."""
    text = DOC.read_text(encoding="utf-8")
    block = text.split("## 2. Licence and cost", 1)
    assert len(block) == 2, "the licence section is gone from docs/why-this-engine.md"
    # Bounded to the section. Unbounded, this walked on into the claims table
    # and demanded a licence link from every row there -- a test that fails for
    # the wrong reason is worse than none, because the fix looks like editing
    # the page.
    section = block[1].split("\n## ", 1)[0]
    unlinked, examined = [], 0
    for line in section.splitlines():
        if not line.startswith("|") or "Licence" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or set("".join(cells)) <= set("-: "):
            continue
        engine, licence, _permits = cells
        examined += 1
        if licence.lower() in ("commercial", ""):
            continue                        # "negotiate" quotes nothing
        if not MARKDOWN_LINK.search(licence):
            unlinked.append(f"{engine}: {licence!r} has no link to the licence text")
    # A floor, like the sibling checks have. Without it this passed on an empty
    # table: deleting every row from the licence section left it green, and so
    # did any reformat that changed the column count. #51's guarantee cannot
    # rest on a loop that may run zero times.
    assert examined >= 5, (
        f"only {examined} licence rows examined; the table has been reformatted "
        f"or emptied. This test passing means nothing until it reads them."
    )
    assert not unlinked, (
        "licence claims about other projects must link to their own documents "
        "(#51):\n  " + "\n  ".join(unlinked)
    )


def test_every_competitor_says_what_it_does_well_and_when_to_choose_it():
    """#52: name each, and say plainly which a reader should pick instead."""
    text = DOC.read_text(encoding="utf-8")
    block = text.split("## 3. Which competitor is actually closest", 1)
    assert len(block) == 2, "the competitor section is gone"
    section = block[1].split("\n## ", 1)[0]
    headings = re.findall(r"^### (.+)$", section, re.MULTILINE)
    assert len(headings) >= 3, f"only {len(headings)} competitors discussed: {headings}"
    for heading in headings:
        # Split on the separator and require text after it. `"-" in heading` was
        # satisfied by any hyphenated name -- `### KuzuDB-lite` passed with no
        # verdict at all, so #52 was only enforced against headings that
        # happened not to contain a hyphen.
        parts = re.split(r"\s+[—-]\s+", heading, maxsplit=1)
        assert len(parts) == 2 and parts[1].strip(), (
            f"competitor heading {heading!r} states no verdict. #52 asks for "
            f"what each does well and where it beats us, in the heading a "
            f"skimmer reads -- `### Name — what it wins`."
        )


@pytest.mark.parametrize("issue", ["#43", "#51", "#52"])
def test_the_document_is_still_anchored_to_its_issues(issue):
    """Each section says which question it answers, so the page is auditable."""
    text = DOC.read_text(encoding="utf-8")
    assert issue in text, (
        f"{issue} is no longer referenced in docs/why-this-engine.md. The page "
        f"exists to answer it; if that changed, this test should change with it."
    )
