"""The few facts _quarto.yml repeats by hand agree with _variables.yml, and every page the
generated navigation links to exists and is rendered."""

from pathlib import Path

import yaml

import common
import gen_tables

V = common.load_variables()
ROOT = common.ROOT
QUARTO = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
RENDER = QUARTO["project"]["render"]


def rendered(page: str) -> bool:
    page = page.lstrip("/")
    return any(Path(page).match(pattern) or page == pattern for pattern in RENDER)


def days_menu() -> list[dict]:
    left = QUARTO["website"]["navbar"]["left"]
    return next(item["menu"] for item in left if item.get("text") == "Days")


def test_the_days_menu_matches_the_day_pages():
    """One item per day page, in day order, reading days.dN.label + ' · ' + days.dN.title."""
    pages = [
        (gen_tables.day_page(V, k).lstrip("/"), f"{d['label']} · {d['title']}")
        for k, d in gen_tables.days(V)
        if gen_tables.day_page(V, k)
    ]
    prework = "modules/00-prework.qmd"  # the pre-work has no day page; its module stands in
    menu = [(item["href"], item["text"]) for item in days_menu() if item["href"] != prework]
    assert menu == pages


def test_every_day_page_and_module_page_exists_and_is_rendered():
    pages = [gen_tables.day_page(V, k) for k, _ in gen_tables.days(V)]
    pages += [gen_tables.module_link(k, m) for k, m in V["modules"].items()]
    for page in filter(None, pages):
        assert (ROOT / page.lstrip("/")).exists(), f"{page} is linked but missing"
        assert rendered(page), f"{page} is not in _quarto.yml project.render"
