# SigmaTau CLI (`bin/sigmatau`) Reference

The `sigmatau` launcher starts the Julia `SigmaTauCLI` app in interactive mode.

## Start

```bash
bin/sigmatau
```

## Command syntax rules

- Positional-first commands: `verb [args...] [--flags]`
- Flags accepted in either form:
  - `--key value`
  - `--key=value`
- Boolean flags (no value):
  - `--open`, `--view`, `--help`, `--verbose`
- Dataset-result selector syntax in `view`/`export`:
  - `NAME:DEV` (example: `clkA:adev`)

## `--m` list grammar

- Explicit list: `--m "1,2,4,8,16"`
- Geometric expansion with endpoint: `--m "1,2,4,...,4096"`
  - Ratio is inferred from the two numbers before `...`

## Commands

| Command | Syntax | Description |
| :--- | :--- | :--- |
| `load` | `load <file> [as <name>] [--col N] [--tau0 S] [--type phase\|freq]` | Load a dataset into session memory. |
| `list` | `list` | Show loaded datasets and stored results. |
| `use` | `use <name>` | Set active dataset. |
| `info` | `info [<name>] [--dev DEV]` | Print dataset metadata or a result summary. |
| `dev` | `dev <name\|all> [...] [--on NAME] [--m "1,2,4,..."]` | Compute one or more deviations and cache results. |
| `noise-id` | `noise-id [<name>] [--dev DEV]` | Print α classification table from stored result. |
| `view` | `view [NAME:DEV ...] [--save PATH.png] [--open]` | Plot one or more stored deviation results. |
| `export` | `export [NAME:DEV] [--out PATH.csv]` | Save stored result to CSV. |
| `help` | `help [<command>]` | Print global or per-command help. |
| `history` | `history` | Show entered commands. |
| `clear` | `clear` | Clear terminal output. |
| `exit` / `quit` | `exit` | Leave session. |

## Common interactive session

```text
load reference/validation/stable32gen.DAT as s32 --tau0 1.0
list
dev adev mdev --on s32 --m "1,2,4,...,1024"
view s32:adev s32:mdev --save results/s32_adev_mdev.png
export s32:adev --out results/s32_adev.csv
```

## Outputs

- In-memory dataset cache per session.
- Stored deviation results keyed by `(dataset, method)`.
- Optional PNG plots via `view --save`.
- CSV export via `export --out`.

## Troubleshooting

- `unknown command`: run `help` and check spelling.
- `no stored result`: run `dev ...` before `view`/`export`.
- `missing value for flag`: ensure value-bearing flags use `--key value` or `--key=value`.
