# MLflow — Step by Step Guide

**Project:** `exam_score_project` — Feature Engineering & MLOps, Semester VII
**Tool in focus:** MLflow only. DVC and GitHub Actions are deliberately left out of this copy of
the project so the MLflow workflow is easy to see in isolation. You will get a DVC version and a
GitHub Actions version separately, and combine all of them yourself later.

**Baseline model (unchanged by MLflow):** the same `sklearn` pipeline from
`notebooks/exam_score_pipeline.ipynb` — `FeatureCreator → ColumnTransformer (numeric scaling,
mock-test PCA, city one-hot, income ordinal) → SelectFromModel(Lasso) → LinearRegression`.
**Baseline metrics:** Test R² = 0.807, Test MAE = 3.91.

---

## 1. WHY — What problem does MLflow solve?

Ask your students this question first:

> "You tried five different `lasso_alpha` values last week while tuning the model. Which one
> gave the best R²? What were the other four numbers?"

Without tracking, the honest answer is usually "I don't remember, it's somewhere in my terminal
scrollback." Every run's parameters, metrics, and the model file itself get overwritten by the
next run, or scattered across print statements nobody saved.

**Simple analogy:** Training a model without experiment tracking is like doing lab experiments
without a lab notebook — you get a result, but nothing records *which* conditions produced it,
so you can't compare experiments or reproduce the best one later. MLflow **is** the lab notebook:
every run is an automatically dated, permanent entry recording what you changed (parameters),
what happened (metrics), and the actual product of the experiment (the model file).

---

## 2. WHAT — The four pieces of MLflow used here

| Concept | What it means | Analogy |
|---|---|---|
| **Experiment** | A named group of related runs (here: `exam_score_prediction`) | A lab notebook's chapter |
| **Run** | One execution of training, with its own params/metrics/artifacts | One dated lab-notebook entry |
| **Tracking store** | Where run metadata (params, metrics, tags) is saved | The notebook's pages |
| **Model Registry** | A separate catalogue of *promoted* models, with named versions and aliases (e.g. "champion") | The shelf of results you've decided are worth keeping and reusing |

**Important distinction for students:** logging a run does *not* register it. Every training run
becomes a tracked run automatically. Only a run you explicitly promote becomes a numbered,
named model version in the Registry — the two are related but separate steps, on purpose:
you try many things, but only register the ones worth keeping.

---

## 3. HOW — The full step-by-step walkthrough (verified, run in order)

> Every command below was actually executed in this project. Outputs shown are real.

### Step 3.1 — Point MLflow at a SQLite-backed tracking store

By default, `mlflow.log_param(...)` etc. write to a plain local folder (`./mlruns/`) with no
database behind it. That works for basic tracking, but the **Model Registry** (Step 3.4) needs a
real database backend to store registered-model metadata. So the very first line of
`src/train_mlflow.py` sets:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("exam_score_prediction")
```

`sqlite:///mlflow.db` creates a single-file SQLite database in the project folder — no server to
install, nothing to configure, perfect for a classroom laptop. In a real company this same line
would instead point at a shared MLflow tracking server (a URL, `http://mlflow.company.com`) so
the whole team's runs land in one place.

> **Gotcha to flag for students:** even with the SQLite tracking URI, MLflow still stores the
> actual model *artifact files* (the pickled pipeline) on local disk under `./mlruns/<experiment
> id>/...` by default — only the metadata (params, metrics, tags, registry entries) goes into
> `mlflow.db`. This split — small structured metadata in a database, large binary artifacts in
> file storage — is the exact same pattern DVC uses (Git for pointers, DVC remote for bytes), and
> it's worth pointing this parallel out explicitly if you've already taught the DVC module.

### Step 3.2 — Instrument the training script

Nothing about the feature engineering or model changes. Three additions wrap the *exact same*
pipeline from the notebook:

```python
with mlflow.start_run(run_name=args.run_name):
    # 1. Log every input that could change the result
    mlflow.log_param("lasso_alpha", args.lasso_alpha)
    mlflow.log_param("pca_components", args.pca_components)
    # ... (test_size, random_state, row counts)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 2. Log the outcome
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # 3. Log the model itself as an artifact
    mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")
```

> **Gotcha to flag for students:** `mlflow.sklearn.log_model(...)` defaults to a newer "skops"
> serialization format, which refuses to load pipelines containing **custom classes** (our
> `FeatureCreator`) without an extra trust flag at load time. We pass
> `serialization_format="pickle"` to match the plain-pickle behaviour `joblib` already uses
> everywhere else in this project, so saving/loading behave consistently. This is the same
> "custom class must live in an importable file" lesson from `features.py`'s own docstring —
> MLflow's serializer hits the identical problem, just one layer further downstream.

Run it once with the baseline parameters:

```bash
PYTHONPATH=. python src/train_mlflow.py --lasso-alpha 0.3 --run-name "alpha_0.3_baseline"
```

```
Test R^2: 0.807
Test MAE: 3.91
Logged to MLflow run: 40fa05863c3e41ae8eb79150c3e71b0a
Experiment: exam_score_prediction
```

Same `PYTHONPATH=.` gotcha as before — `python src/train_mlflow.py` puts `src/`, not the project
root, on the import path, so `from features import FeatureCreator` fails without it.

### Step 3.3 — Run multiple experiments to build something worth comparing

Tracking one run isn't interesting — the value shows up once you have several to compare. Try a
few different `lasso_alpha` values (the same knob explored in the DVC demo, tracked a different
way here):

```bash
PYTHONPATH=. python src/train_mlflow.py --lasso-alpha 0.6 --run-name "alpha_0.6"
PYTHONPATH=. python src/train_mlflow.py --lasso-alpha 0.1 --run-name "alpha_0.1"
```

```
alpha_0.6 -> Test R^2: 0.809   Test MAE: 3.84
alpha_0.1 -> Test R^2: 0.805   Test MAE: 3.90
```

Now compare every run in the experiment, sorted by R², straight from Python (or use the UI —
Step 3.5):

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.get_experiment_by_name("exam_score_prediction")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.r2 DESC"])
print(runs[["tags.mlflow.runName", "params.lasso_alpha", "metrics.r2", "metrics.mae"]])
```

```
 tags.mlflow.runName  params.lasso_alpha  metrics.r2  metrics.mae
            alpha_0.6                 0.6    0.809265     3.835435
   alpha_0.3_baseline                 0.3    0.807276     3.907702
            alpha_0.1                 0.1    0.804527     3.901254
```

`alpha=0.6` wins on both metrics — this is the run we'll promote in the next step.

### Step 3.4 — Register the best run and mark it "champion"

Re-run with `--register` to also save this run as a numbered version in the **Model Registry**:

```bash
PYTHONPATH=. python src/train_mlflow.py --lasso-alpha 0.6 --run-name "alpha_0.6" --register
```

```
Successfully registered model 'exam_score_predictor'.
Created version '2' of model 'exam_score_predictor'.
```

Every `--register` run creates a new, permanent, numbered version (version 1 was the earlier
`alpha_0.3_baseline` run, version 2 is this `alpha_0.6` run). Numbers never get reused or
overwritten — this is exactly the version history a plain `models/exam_score_pipeline.joblib`
file on disk does *not* give you, since a new `joblib.dump()` silently overwrites the old file.

Now mark version 2 as the model actually in use, using an **alias** rather than a hard-coded
version number:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(name="exam_score_predictor", alias="champion", version=2)
```

```
Alias 'champion' set on version 2
```

**Why an alias instead of always writing "version 2" in your deployment code?** Because next
month version 5 might beat it. Code that loads `models:/exam_score_predictor@champion` never
needs to change — you just move the alias to a new version number after the next promotion. This
is precisely the same idea as a `latest` Docker tag, applied to models instead of images.

### Step 3.5 — Explore visually with the MLflow UI

Everything above is also visible in a web UI, which is normally how a team actually browses runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This starts a local server (default `http://127.0.0.1:5000`) showing:

- the **Experiments** tab — every run, sortable/filterable by any logged parameter or metric,
  exactly like the `search_runs()` table above but interactive, with charts comparing runs
- the **Models** tab — `exam_score_predictor` with its 3 versions, and the `champion` alias badge
  visibly attached to version 2

This is the view to project on a classroom screen — students immediately understand "compare
runs" once they see the sortable table with `lasso_alpha`, `r2`, and `mae` as columns.

### Step 3.6 — Verify the round trip: load the champion and predict

The real test of a Model Registry is whether a *different* process — a deployment script, an
API, a teammate's notebook — can load "the current best model" without knowing anything about
how it was trained:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/exam_score_predictor@champion")

preds = model.predict(X.head(5))
```

```
Loaded champion model: <class 'mlflow.pyfunc.PyFuncModel'>
Sample predictions on 5 rows:
[77.70935846 84.3759829  92.29161497 74.97027959 94.96559596]
```

No file path, no pickle-loading code, no knowledge of `lasso_alpha=0.6` anywhere in this
snippet — `models:/exam_score_predictor@champion` is a complete, self-describing address for
"whichever model we currently trust," resolved by MLflow at load time.

---

## 4. SIMPLE EXAMPLE — the mental model in one picture

```
 mlflow.start_run()
        |
        v
 +-------------------------+        +----------------------------+
 |  TRACKING (every run)   |        |  REGISTRY (promoted runs)   |
 |--------------------------|       |------------------------------|
 | run: alpha_0.3_baseline  |       |  exam_score_predictor         |
 |   params: lasso_alpha=0.3|       |    v1  <- alpha_0.3_baseline  |
 |   metrics: r2=0.807      |       |    v2  <- alpha_0.6  @champion|
 | run: alpha_0.6           | --->  |    v3  <- alpha_0.1           |
 |   params: lasso_alpha=0.6|       +----------------------------+
 |   metrics: r2=0.809      |                    ^
 | run: alpha_0.1           |                    |
 |   params: lasso_alpha=0.1|          only runs you explicitly
 |   metrics: r2=0.805      |          --register get promoted here
 +-------------------------+
```

Every run lands in Tracking automatically. Only the ones worth keeping get promoted to the
Registry — and only one of those, at a time, wears the "champion" badge that deployment code
actually reads.

---

## 5. INDUSTRY EXAMPLE

**Fraud detection team at a payments company.** Ten data scientists are all iterating on a fraud
model simultaneously, each trying different feature sets and hyperparameters. Without shared
tracking, comparing "whose approach is actually best" means everyone screenshotting their own
terminal output into a shared doc — slow, error-prone, and impossible to audit later. With a
shared MLflow tracking server:

- every run from every team member lands in the same `exam_score_prediction`-style experiment,
  automatically comparable by metric in one UI
- the on-call ML engineer promotes the winning run to the Registry and points the `champion`
  (or `production`) alias at it
- the serving system always loads `models:/fraud_model@champion` — deploying a better model
  later means moving the alias, not redeploying code

---

## 6. COMMON MISTAKES

| Mistake | Why it's a problem | Fix |
|---|---|---|
| Using the default file-based tracking store, then trying to use the Model Registry | The Registry needs a database backend; the plain file store cannot register models | Set a database-backed `tracking_uri`, e.g. `sqlite:///mlflow.db`, from the start |
| Assuming `mlflow.log_model(...)` automatically registers the model | Logging and registering are separate steps on purpose | Pass `registered_model_name=...` (or call `mlflow.register_model` separately) only for runs you actually want to promote |
| Hard-coding a version number (`models:/exam_score_predictor/2`) in deployment code | Every promotion then requires a code change and redeploy | Use an alias (`models:/exam_score_predictor@champion`) and move the alias instead |
| Letting `mlflow.sklearn.log_model` use its default serialization format with a custom transformer class | The newer default format ("skops") refuses to load custom classes without an explicit trust flag | Pass `serialization_format="pickle"` when the pipeline contains custom classes like `FeatureCreator` |
| Forgetting `PYTHONPATH=.` when running `python src/train_mlflow.py` directly | `from features import FeatureCreator` fails — `src/` is on the path, not the project root | Run as `PYTHONPATH=. python src/train_mlflow.py ...` |

---

## 7. PRODUCTION CONNECTION

This single-tool demo maps directly onto the full MLOps lifecycle you're learning:

```
TRAIN --mlflow.start_run()--> TRACKED RUN (params, metrics, model artifact)
   --compare runs (UI or search_runs)--> PICK THE BEST RUN
   --register_model / --register--> REGISTRY VERSION
   --set_registered_model_alias("champion")--> PROMOTED MODEL
   --mlflow.pyfunc.load_model("models:/name@champion")--> SERVED IN PRODUCTION
```

- **Feature engineering connection:** every parameter that changes a feature (like
  `pca_components`) is logged as an MLflow param, so a later question like "did adding the second
  PCA component actually help?" is answered by one sorted table, not a guess.
- **DVC connection (your other demo):** DVC versions *data and pipelines* so a training run is
  reproducible from source; MLflow versions *what happened when you ran it* and *which result you
  trusted enough to deploy*. In the collated project you'll build later, `dvc repro` reproduces
  the run, and the same run also gets logged to MLflow — reproducibility and experiment
  comparison, together.
- **CI/CD connection (GitHub Actions, your third demo):** a CI pipeline can run training and log
  to MLflow automatically on every push, turning "did this change improve the model?" into an
  automated, recorded answer instead of a manual one.
- **Deployment connection:** the Docker/Streamlit/AWS deployment you already built currently loads
  `models/exam_score_pipeline.joblib` directly from disk. The natural next step (shown here, not
  yet wired into that deployment) is to have the serving code call
  `mlflow.pyfunc.load_model("models:/exam_score_predictor@champion")` instead — so "deploy a
  better model" becomes "move an alias," not "rebuild a Docker image."

---

## 8. Quick Command / API Reference

| Call | What it does |
|---|---|
| `mlflow.set_tracking_uri("sqlite:///mlflow.db")` | Point MLflow at a database-backed tracking store |
| `mlflow.set_experiment(name)` | Create/select the experiment new runs belong to |
| `mlflow.start_run(run_name=...)` | Begin a tracked run (use as a `with` block) |
| `mlflow.log_param(key, value)` | Record one input/hyperparameter for this run |
| `mlflow.log_metric(key, value)` | Record one outcome/metric for this run |
| `mlflow.sklearn.log_model(model, name=..., serialization_format="pickle")` | Save the trained model as a run artifact |
| `registered_model_name=...` (on `log_model`) | Also register this run's model as a new Registry version |
| `MlflowClient().set_registered_model_alias(name, alias, version)` | Point a named alias (e.g. `"champion"`) at a specific version |
| `mlflow.pyfunc.load_model("models:/<name>@<alias>")` | Load whichever version an alias currently points to |
| `mlflow.search_runs(experiment_ids=[...], order_by=[...])` | Query/compare runs programmatically |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Launch the interactive web UI |

---

## 9. Exercise for Students

1. Run `src/train_mlflow.py` with three more `--lasso-alpha` values of your choice. Use
   `mlflow.search_runs(...)` (or the UI) to find the best one by MAE instead of R² — is it the
   same run?
2. Register your best new run, then move the `champion` alias to it. Confirm with
   `mlflow.pyfunc.load_model("models:/exam_score_predictor@champion")` that predictions actually
   change compared to the current champion.
3. Open `mlflow ui --backend-store-uri sqlite:///mlflow.db` and take a screenshot of the
   Experiments table sorted by R². Which single run would you deploy, and why — R² alone, or
   R² and MAE together?
