"""Generated notebook footers and headers keep the core week and the optional extensions
apart: the core never leads into an extension, the capstone says it ends the core week and
points to the extensions page, and each extension leads to the next one. Footers are outside
content_sha, so changing them never makes a run record stale."""

import common
import gen_notebooks
import run_records

V = common.load_variables()
CURRENT = run_records.current_hashes(V)
CORE_NB = [e for e in common.notebooks(V) if e["track"] == common.CORE]
EXT_NB = [e for e in common.notebooks(V) if e["track"] != common.CORE]


def test_the_core_never_leads_into_the_extensions():
    for e in CORE_NB:
        nxt = gen_notebooks.next_notebook(V, e)
        assert nxt is None or nxt["track"] == common.CORE, e["id"]


def test_the_last_core_notebook_says_so_and_points_to_the_extensions():
    footer = gen_notebooks.footer_source(V, CORE_NB[-1])
    assert "**This is the last notebook of the core workshop.**" in footer
    assert "**Next:**" not in footer
    site = V["repo"]["site"].rstrip("/")
    for dkey in common.extension_days(V):
        assert f"({site}/{V['days'][dkey]['track']}.html)" in footer


def test_extension_footers_lead_to_the_next_extension():
    for e, nxt in zip(EXT_NB, EXT_NB[1:], strict=False):
        footer = gen_notebooks.footer_source(V, e)
        assert "**Next extension:** " in footer and f"Module {nxt['n']} · " in footer
    assert "last extension notebook" in gen_notebooks.footer_source(V, EXT_NB[-1])


def test_an_extension_notebook_header_says_it_is_optional():
    header = gen_notebooks.header_source(V, EXT_NB[0])
    assert "(optional, beyond the core week)" in header and "pre-work" not in header


def test_the_footer_is_outside_content_sha():
    """Footers changed with the extension track; no run record may go stale for that."""
    for e in (CORE_NB[0], CORE_NB[-1]):
        nb = gen_notebooks.build(V, e)
        before = common.sha_of_cells(nb.cells)
        footer = next(c for c in nb.cells if c["id"] == common.FOOTER_ID)
        footer["source"] = "something else entirely"
        assert common.sha_of_cells(nb.cells) == before == CURRENT[e["id"]][0]
