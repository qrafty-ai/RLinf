# Docs Knowledge Base

**Generated:** 2026-02-16 05:07:36 UTC
**Commit:** 3d13b38
**Branch:** dev

## OVERVIEW
Documentation is built with Sphinx once per locale; the English docs sit under `docs/source-en/`, the Chinese mirror lives under `docs/source-zh/`, and both share the same `autobuild.sh`/Makefile helpers.

## STRUCTURE
```
docs/
├── source-en/        # primary rst sources + conf.py (English)
├── source-zh/        # translated rst + conf.py (Chinese)
├── autobuild.sh      # live preview + rebuild watcher for both locales
├── Makefile + make.bat # standard Sphinx commands
└── README.md         # build instructions and release notes
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a page or tutorial | `docs/source-en/rst_source/` + `source-zh/rst_source/` | Mirror the same filename for Chinese translations; new rst files must be referenced from `index.rst` to appear on the navbar. |
| Update configuration | `docs/source-*/conf.py` | Both configs set `project = "RLinf"`, `extensions = ["sphinx.ext.autodoc", ...]`, and share a `sys.path` prep that adds the repo root. |
| Preview docs locally | `docs/autobuild.sh` | This script watches both directories and rebuilds HTML; run it in a separate terminal after `source .envrc`. |

## CONVENTIONS
- Write rst using the project’s tone: short intro paragraphs, `.. automodule::` when exposing APIs, and `.. include::` for repeated snippets (e.g., `docs/source-en/snippets/`).\
- Keep the English and Chinese sources synchronized; when adding a page in `source-en`, add a translated version in `source-zh` with the same filename and refer to the bilingual menu defined in their shared `_toc`.\
- The Makefile targets (`html`, `linkcheck`, `gettext`) are invoked from both `autobuild.sh` and CI; avoid editing `_build/` or committing generated HTML.\

## ANTI-PATTERNS
- Do not commit the `_build/` directory or generated PDFs (`docs/_build/`)—the repo relies on source rst only.\
- Do not introduce English/Chinese drift; new docs must pass both locales’ navigation config checks.\

## COMMANDS
```
cd docs && source ../.envrc && make html
cd docs && make clean && make linkcheck
bash docs/autobuild.sh
```

## NOTES
- The README in `docs/` describes the `pixi`/`uv` prereqs for sphinx and outlines how CI uploads the generated docs.\
- Translation PRs should update both locales and rerun `bash docs/autobuild.sh` (watcher) or `make html` before merging.
