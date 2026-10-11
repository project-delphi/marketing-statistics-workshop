"""A Colab link points at repo.ref, so it opens only a notebook that repo.ref contains. A
notebook added after the tag is declared `in_ref: false` in _variables.yml; nothing generated
links it to Colab, every place a Colab link would be shows the "Colab after the next release"
chip (pages) or note (notebook header) pointing at the local setup instead, and removing the
flag brings the links back. A flagged notebook blocks the release of its track. The flag is a
declared fact, so generated files never depend on which tags a checkout has fetched; the last
test checks it against git where repo.ref resolves (CONTRIBUTING.md, "Notebooks a release
does not contain yet")."""

import copy
import datetime as dt
import re
import subprocess

import pytest

import common
import gen_notebooks
import gen_tables as g
import release_check
import run_records

V = common.load_variables()
REF = str(V["repo"]["ref"])
SITE = V["repo"]["site"].rstrip("/")
CURRENT = run_records.current_hashes(V)
CHIP = g.not_in_ref(V)


def colab(v: dict, entry: dict) -> str:
    return f"{v['repo']['colab_base']}/{REF}/{entry['path']}"


def at_ref(entry: dict) -> str:
    """Any link to the notebook at repo.ref (Colab, raw or blob) contains this."""
    return f"/{REF}/{entry['path']}"


def flag(v: dict, ids: set[str], in_ref: bool) -> dict:
    """A copy of v with these notebooks declared in repo.ref (key removed) or not."""
    v = copy.deepcopy(v)
    for m in v["modules"].values():
        for nb in m.get("notebooks") or []:
            if common.notebook_id(nb["path"]) in ids:
                if in_ref:
                    nb.pop("in_ref", None)
                else:
                    nb["in_ref"] = False
    return v


def generated_text(v: dict) -> dict[str, str]:
    """Every file gen_tables writes and every notebook gen_notebooks builds from `v`, as
    {name: text} (a notebook as its cells' sources)."""
    out = {p.relative_to(common.ROOT).as_posix(): t for p, t in g.outputs(v).items()}
    for e in common.existing_notebooks(v):
        out[e["path"]] = "\n".join(c["source"] for c in gen_notebooks.build(v, e).cells)
    return out


def entry_in(v: dict, e: dict) -> dict:
    """The same notebook's entry as `v` declares it (its own in_ref)."""
    return common.notebook_by_id(v)[e["id"]]


# ---- the declaration ---------------------------------------------------------------------------


def test_in_ref_is_only_ever_written_as_false():
    for m in V["modules"].values():
        for nb in m.get("notebooks") or []:
            if "in_ref" in nb:
                assert nb["in_ref"] is False, (
                    f"{nb['path']}: in_ref is {nb['in_ref']!r}; only `in_ref: false` is"
                    " written (remove the key once repo.ref contains the notebook)"
                )


def test_the_chip_points_at_a_section_that_exists():
    page, anchor = common.LOCAL_SETUP.split("#")
    assert re.search(rf"\{{#{anchor}\}}", (common.ROOT / page).read_text(encoding="utf-8"))
    assert f"(/{common.LOCAL_SETUP} " in CHIP and CHIP.endswith("{.chip}")


def test_the_extensions_page_names_the_chip_it_shows():
    text = (common.ROOT / "extensions.qmd").read_text(encoding="utf-8")
    assert f'"{common.NOT_IN_REF}"' in text
    assert '"in preparation" until its notebook exists' not in text


# ---- flagged notebooks: no Colab link anywhere, the chip where the link was -------------------


def flagged_case() -> tuple[dict, list[dict]]:
    """(variables, the written notebooks they flag): the real declaration while it flags any;
    once a release has removed every flag, a copy that flags the last written notebook, so
    these tests keep testing the rule."""
    v = V
    if all(e["in_ref"] for e in common.existing_notebooks(V)):
        v = flag(V, {common.existing_notebooks(V)[-1]["id"]}, in_ref=False)
    return v, [e for e in common.existing_notebooks(v) if not e["in_ref"]]


def test_no_generated_link_points_at_a_flagged_notebook_at_the_ref():
    v, flagged = flagged_case()
    for name, text in generated_text(v).items():
        for e in flagged:
            assert at_ref(e) not in text, f"{name} links {e['path']} at {REF}"


def test_flagged_notebooks_show_the_chip_where_the_colab_link_was():
    v, flagged = flagged_case()
    for e in flagged:
        key, lang = e["module"], g.LANG[e["language"]]
        assert f"{CHIP} [{lang}]{{.chip}}" in g.module_header(v, key)
        assert CHIP in g.lab_block(v, key)
        assert f"{CHIP} [{lang}]{{.chip}}" in g.module_cards(v, v["modules"][key]["day"])
        stem = f"`{e['path'].rsplit('/', 1)[-1].removesuffix('.ipynb')}`"
        table = g.notebooks_table(v, [], core=e["track"] == common.CORE)
        row = next(r for r in table.splitlines() if stem in r)
        assert CHIP in row and "colab-badge" not in row
        cell = gen_notebooks.header_source(v, e)
        assert f"**{common.NOT_IN_REF}.**" in cell and "colab-badge" not in cell
        assert f"({SITE}/{common.LOCAL_SETUP.replace('.qmd', '.html')})" in cell


def test_the_footer_before_a_flagged_notebook_links_its_module_page():
    v, flagged = flagged_case()
    ids = {e["id"] for e in flagged}
    checked = 0
    for e in common.existing_notebooks(v):
        nxt = gen_notebooks.next_notebook(v, e)
        if nxt is None or nxt["id"] not in ids:
            continue
        footer = gen_notebooks.footer_source(v, e)
        slug = v["modules"][nxt["module"]]["slug"]
        assert f"({SITE}/modules/{slug}.html)" in footer, e["id"]
        assert common.NOT_IN_REF in footer and at_ref(nxt) not in footer
        checked += 1
    assert checked or len(flagged) == 1  # only a first notebook has no footer before it


def test_flagging_a_core_notebook_removes_every_link_to_it_including_the_run_sheet():
    """The rule is not about the extensions: any notebook added after a tag is handled."""
    e = next(x for x in common.existing_notebooks(V) if x["id"] == "python/08-bayesian-mmm")
    v = flag(V, {e["id"]}, in_ref=False)
    for name, text in generated_text(v).items():
        assert at_ref(e) not in text, f"{name} still links {e['path']} at {REF}"
    sheet = g.run_sheet(v, V["modules"]["m08"]["day"])
    assert f"Python lab ([{common.NOT_IN_REF}](/{common.LOCAL_SETUP}))" in sheet
    assert at_ref(e) in g.run_sheet(V, V["modules"]["m08"]["day"])  # unflagged: the link


# ---- removing the flag restores the links -------------------------------------------------------


def test_removing_the_flag_restores_every_colab_link():
    flagged_v, flagged = flagged_case()
    v = flag(flagged_v, {e["id"] for e in flagged}, in_ref=True)
    for e in flagged:
        key, lang, url = e["module"], g.LANG[e["language"]], colab(v, e)
        assert f"[Open the {lang} lab in Colab]({url}){{.btn-colab}}" in g.module_header(v, key)
        assert f"[Open in Colab]({url}){{.btn-colab}}" in g.lab_block(v, key)
        assert url in g.module_cards(v, v["modules"][key]["day"])
        table = g.notebooks_table(v, [], core=e["track"] == common.CORE)
        assert f"[![Open in Colab]({g.BADGE})]({url})" in table
        badge = f"[![Open in Colab]({gen_notebooks.BADGE})]({url})"
        assert badge in gen_notebooks.header_source(v, entry_in(v, e))
    for e in common.existing_notebooks(v):
        nxt = gen_notebooks.next_notebook(v, e)
        if nxt is not None and (common.ROOT / nxt["path"]).exists():
            assert colab(v, nxt) in gen_notebooks.footer_source(v, e), e["id"]
    for name, text in generated_text(v).items():
        assert common.NOT_IN_REF not in text, name


def test_the_flag_changes_no_run_record_hash():
    """Header and footer cells are outside content_sha, and the flag is not in deps_sha: a
    run record made before the flag (or before its removal) stays current."""
    ids = {e["id"] for e in common.existing_notebooks(V)}
    unflagged = flag(V, ids, in_ref=True)
    all_flagged = flag(V, ids, in_ref=False)
    for v in (unflagged, all_flagged):
        assert run_records.current_hashes(v) == CURRENT
    for e in common.existing_notebooks(V):
        a = gen_notebooks.build(unflagged, entry_in(unflagged, e))
        b = gen_notebooks.build(all_flagged, entry_in(all_flagged, e))
        assert common.sha_of_cells(a.cells) == common.sha_of_cells(b.cells) == CURRENT[e["id"]][0]
        differ = {c["id"] for c, d in zip(a.cells, b.cells, strict=True) if c != d}
        assert differ <= {common.HEADER_ID, common.FOOTER_ID}, (e["id"], differ)
        assert common.HEADER_ID in differ, e["id"]  # the badge became the note


# ---- the release check ---------------------------------------------------------------------------


def passing(v: dict) -> list[dict]:
    """A current, passing Colab run and CI run of every written notebook (the records
    tests/test_extensions.py builds)."""
    out = []
    for nb_id, (content, deps) in run_records.current_hashes(v).items():
        e = common.notebook_by_id(v)[nb_id]
        base = {
            "notebook": nb_id,
            "kernel": e["kernel"],
            "date": "2026-10-09",
            "mode": "worked",
            "scope": "notebook",
            "status": "pass",
            "seconds": 100.0,
            "evidence": "test",
            "settings": {},
            "content_sha": content,
            "deps_sha": deps,
            "file": "test.json",
        }
        colab_env = "colab-py" if e["kernel"] == "python3" else "colab-r"
        out.append({**base, "source": "colab", "env": colab_env, "index": 0})
        out.append({**base, "source": "test_notebooks", "env": "ci-amd64", "index": 1})
    return out


def test_a_flagged_notebook_blocks_the_release_of_its_track():
    day = dt.date(2026, 10, 9)
    clear = flag(V, {e["id"] for e in common.notebooks(V)}, in_ref=True)
    records = passing(clear)
    assert release_check.blockers(clear, records, day, extensions=True) == []
    core = flag(clear, {"python/08-bayesian-mmm"}, in_ref=False)
    found = release_check.blockers(core, records, day)
    assert len(found) == 1 and found[0].startswith("python/08-bayesian-mmm: declared `in_ref")
    ext = [e for e in common.existing_notebooks(V) if e["track"] != common.CORE]
    flagged_ext = flag(clear, {e["id"] for e in ext}, in_ref=False)
    assert release_check.blockers(flagged_ext, records, day) == []  # extensions not required
    found = release_check.blockers(flagged_ext, records, day, extensions=True)
    assert {b.split(":", 1)[0] for b in found} == {e["id"] for e in ext}
    assert all("`in_ref: false`" in b for b in found)


# ---- the declaration agrees with git ------------------------------------------------------------


def paths_in_ref(ref: str, paths: list[str]) -> set[str] | None:
    """The paths among `paths` that `ref` contains, or None when git is missing or `ref` does
    not resolve here (a shallow checkout without tags)."""
    git = ["git", "-C", str(common.ROOT)]
    try:
        found = subprocess.run(
            [*git, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if found.returncode != 0:
        return None
    listed = subprocess.run(
        [*git, "ls-tree", "-r", "--name-only", ref, "--", *paths],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(listed.stdout.split())


def test_in_ref_agrees_with_git_where_the_ref_resolves():
    entries = common.notebooks(V)
    present = paths_in_ref(REF, [e["path"] for e in entries])
    if present is None:
        pytest.skip(f"repo.ref {REF} does not resolve in this checkout (no tags fetched)")
    wrong = []
    for e in entries:
        if e["in_ref"] and e["path"] not in present:
            wrong.append(f"{e['path']}: {REF} does not contain it; add `in_ref: false`")
        if not e["in_ref"] and e["path"] in present:
            wrong.append(f"{e['path']}: {REF} contains it; remove `in_ref: false`")
    assert not wrong, "\n".join(wrong)
