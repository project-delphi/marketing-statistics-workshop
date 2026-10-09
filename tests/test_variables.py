"""_variables.yml is consistent with itself and with the files it names. Independent of
_quarto.yml and the site's navigation."""

import re
from pathlib import Path

import common
import gen_tables

V = common.load_variables()
ROOT = common.ROOT


def test_every_module_has_a_day_and_a_slug_that_matches_its_id():
    for key, m in V["modules"].items():
        assert re.fullmatch(r"m\d{2}", key), key
        assert m["day"] in V["days"], key
        assert m["slug"].startswith(key[1:] + "-"), key
        assert set(m["minutes"]) == {"briefing", "lab", "debrief"}, key


def test_every_taught_module_is_scheduled_on_its_day():
    scheduled = {
        b["id"]: dkey
        for dkey, d in V["days"].items()
        for b in d.get("blocks") or []
        if b["kind"] == "module"
    }
    for key, m in V["modules"].items():
        if m.get("optional") and not V["days"][m["day"]].get("blocks"):
            continue  # pre-work: done on your own
        assert scheduled.get(key) == m["day"], f"{key} is not in {m['day']}'s blocks"


def test_days_fit_nine_to_five_and_parts_add_up():
    assert gen_tables.part_problems(V) == []
    for dkey in V["days"]:
        segs = gen_tables.plan(V, dkey)
        if segs:
            assert segs[0]["start"] >= gen_tables.to_minutes("09:00"), dkey
            assert segs[-1]["end"] <= gen_tables.to_minutes("17:00"), dkey


def test_notebook_paths_match_their_sources():
    for e in common.notebooks(V):
        k = common.KERNELS[e["kernel"]]
        assert Path(e["path"]).parent == Path("labs") / k["dir"], e["path"]
        assert Path(e["source"]).suffix == k["ext"], e["source"]
        assert Path(e["path"]).stem == Path(e["source"]).stem, e["path"]
        assert e["path"].startswith(f"labs/{k['dir']}/{e['module'][1:]}-"), e["path"]


def test_package_pins_agree_with_requirements():
    pins = common.requirements_pins()
    for name, p in V["packages"].items():
        assert pins.get(common.normalize(name)) == str(p["version"]), name
        assert p.get("checked") and p.get("docs"), f"{name}: record when and where it was checked"


def test_every_notebook_install_resolves():
    r_image = {
        line.strip()
        for line in (ROOT / "environment" / "r-packages.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for e in common.notebooks(V):
        if e["kernel"] == "python3":
            pins = common.python_pins(V, e)
            reqs = common.requirements_pins()
            for name, version in pins.items():
                assert reqs.get(name) == version, f"{e['id']}: {name} {version} not in requirements"
        else:
            r = common.r_install(V, e)
            missing = set(r["cran"]) - r_image
            assert not missing, f"{e['id']}: {missing} not in environment/r-packages.txt"


def test_deps_exist():
    for key, m in V["modules"].items():
        for pattern in m.get("deps") or []:
            assert list(ROOT.glob(pattern)), f"{key}: deps entry {pattern} matches no file"


def test_data_entries_name_a_source():
    for key, m in V["modules"].items():
        for item in m.get("data") or []:
            if isinstance(item, dict):
                assert item.get("name") and item.get("url"), key
                if "sha256" in item:
                    assert re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])), key


def test_readiness_envs():
    envs = V["readiness"]["envs"]
    for kernel in common.KERNELS:
        teaching = [k for k, e in envs.items() if e.get("teaching") and e.get("kernel") == kernel]
        assert len(teaching) == 1, f"one teaching env for {kernel}"
    for key, e in envs.items():
        assert e.get("label"), key
        assert not (e.get("documented_only") and (e.get("teaching") or e.get("ci"))), key
    assert V["readiness"]["max_run_age_days"] > 0


def test_repo_urls_agree():
    repo = V["repo"]
    path = f"{repo['owner']}/{repo['name']}"
    assert repo["url"].endswith(path)
    assert repo["raw"].endswith(path)
    assert path in repo["colab_base"]
