# GitHub Actions (CI/CD) — Step by Step Guide

**Project:** `exam_score_project` — Feature Engineering & MLOps, Semester VII
**Tool in focus:** GitHub Actions only. DVC and MLflow are deliberately left out of this copy of
the project so the CI workflow is easy to see in isolation. You already have the DVC and MLflow
versions and will combine everything into one project yourself later.

**Baseline model (unchanged by CI):** the same `sklearn` pipeline from
`notebooks/exam_score_pipeline.ipynb`. CI does not retrain or change the model — it protects it,
by running the existing test suite automatically.

**Important, read first:** this sandbox cannot push to a real GitHub repository, so we cannot
show you an actual green checkmark on github.com. What we verified instead is exactly what
GitHub's runner would do: we validated the workflow file's YAML, and we ran the *identical*
commands the workflow runs (`pip install -r requirements.txt`, `python -m pytest tests/ -v`) in
a completely clean, fresh virtual environment — the same thing GitHub's own clean Ubuntu machine
does. Section 3.5 below tells you exactly how to see the real green checkmark once this project
is pushed to your own GitHub repo.

---

## 1. WHY — What problem does GitHub Actions solve?

Ask your students this question first:

> "You changed `features.py`. Did you remember to run the test suite before merging your
> pull request? Did the *other* three people on your team, before merging theirs?"

DVC made *data and pipelines* reproducible. MLflow made *experiments* comparable. Neither one
stops a broken change from being merged — that requires something that runs automatically,
for everyone, every single time, whether or not a human remembers.

**Simple analogy:** GitHub Actions is like a security guard who checks every delivery truck at
the warehouse gate, every time, without needing to be reminded. A human guard who "usually
remembers" to check trucks is not a real control. An automatic gate that checks every truck, no
exceptions, is.

**MLOps vs. DevOps, explained here:** DevOps CI usually asks "does the code compile / pass unit
tests?" MLOps CI asks that too, but also increasingly asks ML-specific questions — "did this
change the model's metrics? is the model artifact still present? did feature engineering logic
break?" This workflow demonstrates the DevOps-style question (tests pass); the exercises at the
end point toward the ML-specific question.

---

## 2. WHAT — The pieces of a GitHub Actions workflow

| Concept | What it means | Analogy |
|---|---|---|
| **Workflow** | One `.yml` file describing an automated process | The guard's entire checklist |
| **Trigger (`on:`)** | What event starts the workflow (a push, a pull request, a schedule...) | What makes the guard walk to the gate |
| **Job** | A group of steps that run together on one virtual machine | One guard's shift |
| **Runner** | The actual (temporary, disposable) virtual machine the job executes on | The gate booth itself |
| **Step** | One command or reusable action inside a job | One item on the guard's checklist |
| **Action (`uses:`)** | A pre-built, reusable step someone else wrote (e.g. "check out this repo") | A tool the guard already has, instead of building one from scratch |

---

## 3. HOW — The full step-by-step walkthrough (verified, run in order)

> Every command below was actually executed against this project. Outputs shown are real.

### Step 3.1 — Confirm the baseline passes locally, in a clean environment

Before adding any automation, prove the thing you're about to automate actually works, the same
way GitHub's runner will run it — a brand-new virtual environment, not your existing one that
might have extra packages installed from other work:

```bash
python3 -m venv /tmp/ci_sim_venv
source /tmp/ci_sim_venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

```
tests/test_api.py::test_predict_response_is_in_a_sane_range PASSED       [ 25%]
tests/test_api.py::test_predict_matches_pipeline_prediction_directly PASSED [ 31%]
...
tests/test_pipeline.py::test_irrelevant_columns_do_not_change_prediction_much PASSED [100%]
======================== 16 passed, 2 warnings in 2.13s ========================
```

This is exactly what the workflow below automates — nothing more.

### Step 3.2 — Write the workflow file

GitHub looks for workflow files in one specific, fixed location:
`.github/workflows/<any-name>.yml`. This project's is `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Run test suite
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest tests/ -v

      - name: Verify the trained model artifact is present
        run: |
          if [ ! -f "models/exam_score_pipeline.joblib" ]; then
            echo "::error::models/exam_score_pipeline.joblib is missing from the repo."
            exit 1
          fi
          echo "Model artifact present: models/exam_score_pipeline.joblib"
```

Read the `on:` block first with students: this workflow fires on every push to `main`, **and**
on every pull request targeting `main`. That second trigger is the important one for teamwork —
it means the checklist runs on a proposed change *before* it's merged, not after.

Each `steps:` entry is either `uses:` (run someone else's pre-built action — here, official
GitHub actions that check out the repo and install Python) or `run:` (run a shell command
yourself, exactly like you would in a terminal).

### Step 3.3 — Validate the workflow file itself

You don't need a real GitHub push to catch a YAML mistake — validate the file's syntax locally
first:

```python
import yaml
with open(".github/workflows/ci.yml") as f:
    doc = yaml.safe_load(f)
print("YAML parses OK")
```

```
YAML parses OK
```

> **Real gotcha to flag for students:** if you print the parsed dictionary's keys, the `on:` key
> shows up as the Python boolean `True`, not the string `"on"`. This is a famous YAML 1.1 quirk —
> bare words like `on`, `off`, `yes`, `no` are interpreted as booleans by generic YAML parsers.
> GitHub's own workflow parser is YAML-Actions-aware and handles `on:` correctly regardless, so
> your workflow still runs fine — but if you ever use a generic YAML linter/tool on a workflow
> file and see `on` turn into `True`, this is why, and it's not a bug in your file.

### Step 3.4 — Prove the workflow actually catches a real bug

The whole point of CI is that it stops a broken change from looking safe to merge. Let's break
something on purpose and watch the *exact* command the workflow runs catch it.

Introduce a realistic typo bug into `features.py` — a mistyped column name:

```diff
- X["tenure_days"] = (pd.Timestamp("2026-01-01") - pd.to_datetime(X["enrollment_date"])).dt.days
+ X["tenure_days"] = (pd.Timestamp("2026-01-01") - pd.to_datetime(X["enrollment_dat"])).dt.days
```

Run the exact same command the workflow's "Run tests" step runs:

```bash
python -m pytest tests/ -v
```

```
FAILED tests/test_api.py::test_predict_with_valid_payload_returns_200
FAILED tests/test_api.py::test_predict_response_has_expected_shape
FAILED tests/test_api.py::test_predict_response_is_in_a_sane_range - KeyError...
FAILED tests/test_api.py::test_predict_matches_pipeline_prediction_directly
FAILED tests/test_api.py::test_predict_accepts_unseen_city_gracefully
FAILED tests/test_pipeline.py::test_pipeline_predicts_a_reasonable_score - Ke...
FAILED tests/test_pipeline.py::test_pipeline_handles_missing_study_hours - Ke...
FAILED tests/test_pipeline.py::test_pipeline_handles_unseen_city - KeyError...
FAILED tests/test_pipeline.py::test_higher_study_hours_predicts_higher_score
FAILED tests/test_pipeline.py::test_pipeline_output_is_deterministic - KeyErr...
FAILED tests/test_pipeline.py::test_irrelevant_columns_do_not_change_prediction_much
=================== 11 failed, 5 passed, 2 warnings in 4.93s ===================
```

One mistyped column name breaks 11 of 16 tests, because `tenure_days` feeds into the numeric
branch of the `ColumnTransformer`, which almost every prediction path touches. On a real GitHub
repo, this exact failure is what a pull request author would see as a **red X** next to their
commit, directly in the PR — before a human reviewer even opens the diff.

Revert the bug and confirm green again:

```bash
git checkout -- features.py   # or manually undo the typo
python -m pytest tests/ -v
```

```
======================== 16 passed, 2 warnings in 2.01s ========================
```

This before/after is the single most important thing to demonstrate live in class: **CI doesn't
prevent bugs from being written — it prevents them from being merged unnoticed.**

### Step 3.5 — See the real thing on GitHub (do this after class, or live if you have a repo ready)

Everything above proves the workflow is correct without needing GitHub itself. To see the actual
green checkmark:

1. Create an empty repository on github.com (do **not** initialize it with a README).
2. From this project folder:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. Open the repository's **Actions** tab on github.com. You'll see the `CI` workflow listed,
   with a run already in progress (triggered automatically by the push you just made) or
   completed with a green check.
4. To see the pull-request trigger in action: create a branch, make a small change, open a pull
   request against `main`. GitHub shows the workflow's status directly on the PR page.

---

## 4. SIMPLE EXAMPLE — the mental model in one picture

```
 developer pushes code  /  opens a pull request
                |
                v
   +-------------------------------------------+
   |  GitHub spins up a FRESH ubuntu-latest VM  |
   |---------------------------------------------|
   |  1. checkout repo                          |
   |  2. install Python 3.11                    |
   |  3. pip install -r requirements.txt        |
   |  4. python -m pytest tests/ -v             |
   |  5. verify models/exam_score_pipeline.joblib exists |
   +-------------------------------------------+
                |
        all steps succeed?  ----YES---->  green check, safe to merge
                |
                NO
                v
          red X on the commit/PR, merge blocked from looking safe
```

---

## 5. INDUSTRY EXAMPLE

**E-commerce recommendation team.** Multiple engineers touch the same feature-engineering
codebase — one is adding a new "days since last purchase" feature, another is refactoring the
`ColumnTransformer`. Without CI, the first sign of a conflict is a customer-facing outage after
deployment. With GitHub Actions gating every pull request:

- every PR must pass the full test suite before it can be merged (enforced with a GitHub branch
  protection rule requiring the `CI` check to pass)
- a broken feature-engineering change is caught in the PR itself, often within minutes of being
  pushed, by whoever opened it — not days later by a teammate or a production incident
- this is the same automated gate that, in a more advanced setup, would also run `dvc repro`
  (your DVC demo) to retrain and check metrics haven't regressed, and log the result to MLflow
  (your MLflow demo) — CI is the glue that can call either tool automatically

---

## 6. COMMON MISTAKES

| Mistake | Why it's a problem | Fix |
|---|---|---|
| Putting the workflow file anywhere other than `.github/workflows/*.yml` | GitHub simply won't discover it — no error, it just never runs | Use the exact path `.github/workflows/<name>.yml` |
| Testing only on `push` and forgetting `pull_request` | Nothing stops a broken PR from *looking* mergeable before it's merged — the whole point of blocking bad merges is lost | Trigger on both `push` and `pull_request` |
| Assuming the CI machine has your local packages already installed | "Works on my machine" — CI runners start completely clean every time | Declare every dependency in `requirements.txt`; never rely on anything pre-installed |
| Not pinning a Python version | A subtle version-dependent bug (like a library default changing) can pass locally and fail in CI, or vice versa, for reasons that look random | Pin `python-version` explicitly, matching what your team develops with |
| Treating a workflow file's YAML the same as any other YAML | Bare `on`/`off`/`yes`/`no` keys parse as booleans in generic YAML tools (though GitHub's own parser handles `on:` correctly) | Know this quirk exists so a linter's output about `on: true` doesn't alarm you |

---

## 7. PRODUCTION CONNECTION

This single-tool demo maps directly onto the full MLOps lifecycle you're learning:

```
CODE CHANGE  --git push / pull request-->  CI TRIGGERS
   --checkout + install + pytest-->  PASS/FAIL SIGNAL
   --branch protection rule-->  MERGE ALLOWED ONLY IF GREEN
   --(a more advanced pipeline would also)-->  dvc repro --> mlflow log --> deploy
```

- **DVC connection (one of your other demos):** a more advanced CI workflow would add a step
  running `dvc repro` and comparing `dvc metrics diff` against `main` — automatically failing the
  PR if a change makes R² drop below some threshold, not just if a unit test fails.
- **MLflow connection (your other demo):** that same advanced workflow could log the CI-run
  training as an MLflow run automatically, so "every commit's exact metrics" become queryable
  later without anyone manually running `mlflow ui`.
- **Deployment connection:** once tests pass in CI, the natural next stage (not built here, but
  the logical next step) is a **CD** (continuous deployment) job — build the Docker image you
  already created, push it to a registry, and redeploy to the AWS EC2 instance automatically,
  turning "merge to main" into "shipped to production" with no manual steps in between.
- **Governance connection:** a required, always-green CI check is itself a piece of audit
  evidence — proof that every change to a production ML system passed a defined quality gate
  before being merged, which is exactly the kind of trail a real MLOps governance review asks for.

---

## 8. Quick Reference

| Piece | What it does |
|---|---|
| `.github/workflows/*.yml` | The only location GitHub scans for workflow files |
| `on: push` / `on: pull_request` | Which Git events trigger this workflow |
| `runs-on: ubuntu-latest` | Which clean virtual machine image the job runs on |
| `uses: actions/checkout@v4` | Pulls your repo's code onto the runner |
| `uses: actions/setup-python@v5` | Installs a specific Python version, with optional pip caching |
| `run: <shell command>` | Runs an arbitrary shell command as a step |
| GitHub repo's **Actions** tab | Where to see run history, logs, and pass/fail status |
| Branch protection rule ("Require status checks to pass") | Turns a green CI check from a suggestion into an enforced merge gate |

---

## 9. Exercise for Students

1. Push this project to your own (new, empty) GitHub repository following Step 3.5, then open
   the Actions tab and confirm you see a real green check.
2. On a new branch, introduce a deliberate bug (try a different one from Step 3.4 — for example,
   change `OrdinalEncoder(categories=[["Low", "Medium", "High"]])` to leave out `"High"`), push
   the branch, and open a pull request against `main`. Screenshot the red X GitHub shows you.
3. In your repo's Settings → Branches, add a branch protection rule on `main` requiring the `CI`
   check to pass before merging. Try to merge your broken PR from step 2 — what happens?
4. (Stretch) Add a second job to `ci.yml` that only runs `dvc repro` (using the DVC demo you
   built separately) and fails the PR if `metrics.json`'s R² drops below 0.75 compared to `main`.
