"""What the generators, the runner and the run records share: _variables.yml, the list of
lab notebooks, the pins each notebook installs, and the two hashes that make a run record
stale (content_sha and deps_sha; runs/README.md)."""

from __future__ import annotations

import glob
import hashlib
import json
import re
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
REQUIREMENTS = ROOT / "environment" / "requirements.txt"

# Cell ids of the five cells scripts/gen_notebooks.py writes around a lab's own cells.
HEADER_ID = "workshop-header"
INSTALL_ID = "workshop-install"
HARNESS_ID = "workshop-harness"
RECORD_ID = "workshop-record"
FOOTER_ID = "workshop-footer"
GENERATED_IDS = (HEADER_ID, INSTALL_ID, HARNESS_ID, RECORD_ID, FOOTER_ID)

KERNELS = {
    "python3": {
        "language": "python",
        "dir": "python",
        "ext": ".py",
        "fmt": "py:percent",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "ir": {
        "language": "r",
        "dir": "r",
        "ext": ".R",
        "fmt": "R:percent",
        "kernelspec": {"name": "ir", "display_name": "R", "language": "R"},
        "language_info": {"name": "R"},
    },
}
# R packages every R lab needs for the helpers in R/mktstats.R.
R_BASE_PACKAGES = ("jsonlite",)
# R packages that need the GSL system library (missing on Colab).
NEEDS_GSL = ("CLVTools",)


def load_variables() -> dict:
    return yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))


def module_number(key: str) -> int:
    """m07 -> 7."""
    return int(key[1:])


def notebook_id(path: str) -> str:
    """labs/python/00-setup-warmup.ipynb -> python/00-setup-warmup (the id in run records)."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def notebooks(v: dict) -> list[dict]:
    """Every notebook declared in modules.*.notebooks, in module order: one dict with the
    module key, id, path, source, kernel, language and install list."""
    out = []
    for key in sorted(v["modules"], key=module_number):
        m = v["modules"][key]
        for i, nb in enumerate(m.get("notebooks") or []):
            k = KERNELS[nb["kernel"]]
            out.append(
                {
                    "module": key,
                    "n": module_number(key),
                    "index": i,
                    "id": notebook_id(nb["path"]),
                    "path": nb["path"],
                    "source": nb["source"],
                    "kernel": nb["kernel"],
                    "language": k["language"],
                    "install": list(nb.get("install") or []),
                }
            )
    return out


def existing_notebooks(v: dict) -> list[dict]:
    return [e for e in notebooks(v) if (ROOT / e["path"]).exists()]


def notebook_by_id(v: dict) -> dict[str, dict]:
    return {e["id"]: e for e in notebooks(v)}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@cache
def requirements_pins() -> dict[str, str]:
    """{normalized name: version} from environment/requirements.txt (markers ignored)."""
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        spec = line.split(";", 1)[0].strip()
        name, version = spec.split("==", 1)
        pins[normalize(name.split("[", 1)[0])] = version.strip()
    return pins


def python_pins(v: dict, entry: dict) -> dict[str, str]:
    """{name: version} the install cell makes sure of: each name in the notebook's
    `install`, at the version in _variables.yml `packages` or else requirements.txt."""
    packages = {normalize(k): str(p["version"]) for k, p in (v.get("packages") or {}).items()}
    reqs = requirements_pins()
    out = {}
    for item in entry["install"]:
        name, _, version = str(item).partition("==")
        name = normalize(name.strip())
        version = version.strip() or packages.get(name) or reqs.get(name)
        if not version:
            raise ValueError(f"{entry['id']}: no pin for {name} in packages or requirements.txt")
        out[name] = version
    return dict(sorted(out.items()))


def r_install(v: dict, entry: dict) -> dict:
    """{cran: [...], github: {name: [repo, sha]}, gsl: bool, p3m_url} for an R notebook."""
    github_all = v.get("github_packages") or {}
    names = list(dict.fromkeys([*R_BASE_PACKAGES, *entry["install"]]))
    github = {n: [github_all[n]["repo"], github_all[n]["sha"]] for n in names if n in github_all}
    cran = [n for n in names if n not in github_all]
    return {
        "cran": cran,
        "github": github,
        "gsl": any(n in NEEDS_GSL for n in names),
        "p3m_url": v["environment"]["p3m_url"],
    }


def pins_text(v: dict, entry: dict) -> str:
    """The pins as one stable string, for deps_sha."""
    if entry["kernel"] == "python3":
        return json.dumps(python_pins(v, entry), sort_keys=True)
    return json.dumps(r_install(v, entry), sort_keys=True)


def _cell_text(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def sha_of_cells(cells: list) -> str:
    """A short hash of a lab's own code cells (the generated cells are skipped)."""
    code = [
        _cell_text(cell)
        for cell in cells
        if cell["cell_type"] == "code" and cell.get("id") not in GENERATED_IDS
    ]
    return hashlib.sha256("\n\x1e\n".join(code).encode("utf-8")).hexdigest()[:16]


def content_sha(path: str | Path) -> str:
    """The content_sha of a committed notebook (a path relative to the repository)."""
    nb = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return sha_of_cells(nb["cells"])


def dep_files(v: dict, module: str, kernel: str) -> list[str]:
    """The files behind modules.<module>.deps that matter to a notebook of this kernel:
    plain paths, globs and directories. R files count only for R notebooks and the Python
    package only for Python notebooks; anything else (data) counts for both."""
    out = []
    for pattern in v["modules"][module].get("deps") or []:
        matches = sorted(glob.glob(str(ROOT / pattern), recursive=True)) or [str(ROOT / pattern)]
        for match in matches:
            p = Path(match)
            files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
            for f in files:
                rel = f.relative_to(ROOT).as_posix()
                if "__pycache__" in rel:
                    continue
                is_r = rel.startswith("R/") or rel.endswith((".R", ".r"))
                is_py = rel.startswith("src/")
                if (is_r and kernel != "ir") or (is_py and kernel != "python3"):
                    continue
                out.append(rel)
    return list(dict.fromkeys(out))


def deps_sha(v: dict, entry: dict) -> str:
    """A short hash of the module's declared deps (file contents), this notebook's pins and
    the install ref. A missing file hashes as missing, so adding it changes the hash."""
    h = hashlib.sha256()
    for rel in dep_files(v, entry["module"], entry["kernel"]):
        path = ROOT / rel
        h.update(rel.encode() + b"\0")
        h.update(path.read_bytes() if path.is_file() else b"<missing>")
        h.update(b"\0")
    h.update(b"pins\0" + pins_text(v, entry).encode())
    h.update(b"\0ref\0" + str(v["repo"]["ref"]).encode())
    return h.hexdigest()[:16]
