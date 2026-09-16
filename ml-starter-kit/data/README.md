# data/

- `raw/` — original, untouched source data. Never edit files here by hand; regenerate them
  with their source script instead (see `data/raw/generate_sample_data.py` for the bundled
  example dataset).
- `processed/` — cleaned/feature-engineered data saved by your own scripts or notebooks, if
  you choose to persist intermediate results. Nothing in this framework writes here
  automatically.

Both folders are gitignored by default (see `.gitignore`) so raw and processed data don't
bloat the repository — commit the *scripts that generate the data*, not the data itself,
wherever practical.
