# From a Trained Model to a Live URL: Docker, Streamlit, Docker Hub, AWS

This guide picks up exactly where the main `README.md` leaves off. You already have: a
trained pipeline (`models/exam_score_pipeline.joblib`) and a working FastAPI service
(`app/main.py`) that you can run with `uvicorn` on your own laptop. That's Day 1. This guide
is Day 2 through "a real person, on their own computer, can open a link and use it" —
without you leaving your laptop running.

**The path, in order (do not skip steps):**

```
1. Docker basics          -> package the API alone into an image, run it, understand it
2. Streamlit app          -> build a UI that talks to the API over HTTP
3. Bundle both            -> ONE image, TWO processes, ONE entrypoint script
4. Docker Hub             -> push that image somewhere anyone (and AWS) can pull it from
5. AWS EC2                -> a real machine on the internet, running your container 24/7
```

Every step has a reason. Skipping straight to step 5 is exactly how students end up with a
container that "sort of works" and no idea why.

---

# PART 1 — Docker, from zero

## WHY

Right now, "run the app" means: *you* have Python 3.11, *you* ran
`pip install -r requirements.txt`, *you* have the exact right versions of pandas/scikit-learn
installed, and *you* remembered to train the model first. Hand your laptop's exact setup to a
teammate, a grader, or a cloud server, and none of that is guaranteed to be true. "Works on my
machine" is not a deployable state.

## WHAT

**Docker** packages your application AND everything it needs to run (Python interpreter,
libraries, your code, your trained model file) into one file called an **image**. Running that
image produces a **container** — an isolated, running copy of exactly that environment,
identical whether it's your laptop, a teammate's laptop, or a server in Virginia.

Think of an image as a sealed shipping container for software: what's inside is fixed at
build time, and any ship (any machine with Docker installed) can carry it without caring
what's inside.

| Term | Plain meaning |
|---|---|
| **Image** | A frozen, read-only snapshot — code + dependencies + OS files. Built once. |
| **Container** | A running instance of an image. You can start/stop/delete many containers from one image. |
| **Dockerfile** | The recipe — a text file of instructions that *builds* an image. |
| **Registry** (e.g. Docker Hub) | Where built images are stored so other machines can `pull` them. |

## HOW — reading `docker/Dockerfile.api-only` line by line

This is deliberately the *simplest possible* version — API only, no Streamlit yet:

```dockerfile
FROM python:3.11-slim          # start from an official, minimal Python 3.11 image
WORKDIR /app                    # every following instruction runs from /app inside the image

COPY requirements.txt .         # copy JUST this file first...
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi "uvicorn[standard]" pydantic
                                 # ...and install dependencies before copying any code

COPY features.py .
COPY app/ app/
COPY models/ models/            # NOW copy the actual application + trained model

EXPOSE 8000                     # documentation: this container listens on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
                                 # the command that runs when a container starts
```

**Why dependencies get copied and installed *before* the application code** — this is a real
Docker idiom, not arbitrary ordering. Docker builds an image in **layers**, one per
instruction, and caches each layer. If `requirements.txt` hasn't changed since your last
build, Docker reuses the cached "installed dependencies" layer instead of reinstalling
pandas/scikit-learn from scratch — which is slow. Your Python code (`app/`, `features.py`)
changes far more often than your dependency list, so putting the slow, rarely-changing step
first means most rebuilds are fast.

## SIMPLE EXAMPLE — build it and run it

```bash
# from the project root
docker build -f docker/Dockerfile.api-only -t exam-score-api:practice .
```

- `-f docker/Dockerfile.api-only` — which Dockerfile to use (the default is just `Dockerfile`
  in the current directory, which we're saving for Part 3's combined image).
- `-t exam-score-api:practice` — **t**ag: a human name (`exam-score-api`) plus a version
  (`practice`), instead of Docker's default random hash.
- `.` — the **build context**: "everything Docker is allowed to `COPY` from is this directory
  and below." This is also why `.dockerignore` exists (below) — the whole `.venv/` folder
  sitting in this directory would otherwise get scanned into the build context too.

Run it:

```bash
docker run -p 8000:8000 exam-score-api:practice
```

`-p 8000:8000` means **host port : container port** — "forward my laptop's port 8000 to port
8000 inside the container." Without `-p`, the container's port 8000 is only reachable from
*inside* the container — nothing on your laptop could reach it.

Test it exactly like you did with plain `uvicorn`:

```bash
curl http://localhost:8000/health
```

Same request, same response — the only thing that changed is *where the process is running*.

## `.dockerignore` — why it exists

```
.venv/
__pycache__/
.git/
notebooks/
tests/
data/raw/*.csv
...
```

Everything listed here is real on your laptop but has no business inside the image: your
virtual environment (the image installs its *own* dependencies fresh), the notebook and raw
CSV (training happens *before* Docker, not inside it — the image only ever receives the
already-trained `.joblib` file), test files, `.git` history. Excluding them keeps the image
smaller, the build faster, and — for `.git` especially — keeps repository history from ending
up inside a shippable artifact.

## INDUSTRY EXAMPLE

A bank's fraud-detection API might depend on a very specific, pinned version of scikit-learn
because a newer version changed floating-point behavior in a way that shifted risk scores by
fractions of a percent — enough to matter at bank scale. Docker is how that team guarantees
the exact same scikit-learn version runs in the data scientist's notebook, the CI test suite,
staging, and production — four different machines, one frozen environment.

## COMMON MISTAKES

- **Forgetting to train the model first.** `docker/Dockerfile.api-only` *copies*
  `models/exam_score_pipeline.joblib` into the image — it doesn't train it. If that file
  doesn't exist on your laptop before you `docker build`, the build fails at `COPY models/
  models/`.
- **Not rebuilding after a code change.** Editing `app/main.py` and re-running `docker run`
  on the *old* image runs the *old* code — you must `docker build` again.
- **`-p` backwards.** `-p 8000:8000` is host:container. Mixing this up is a common source of
  "it built fine but I can't reach it."

## PRODUCTION CONNECTION

This is literally how the exam-score-mlops sibling project's CI/CD pipeline works: a GitHub
Actions workflow builds this exact kind of image on every push, runs the test suite *inside*
a container built from it (so "tests passed" means "tests passed in the same environment
that will actually run in production," not just on the CI runner's own Python), and only then
allows a deploy.

---

# PART 2 — The Streamlit app

## WHY

FastAPI's `/docs` page is great for developers, but nobody wants to type raw JSON to check a
student's exam score. A **user interface** — forms, sliders, buttons — is how a non-technical
person (a teacher, an academic advisor) actually uses this.

## WHAT

**Streamlit** turns a plain Python script into a web app — no HTML, CSS, or JavaScript
required. You write `st.number_input(...)`, `st.button(...)`; Streamlit renders it as a
browser page and re-runs your script top-to-bottom on every interaction.

## HOW — the one design decision that matters

Open `streamlit_app.py`. Notice what it does **not** do: it never calls
`joblib.load("models/exam_score_pipeline.joblib")`. It only ever sends an HTTP request to the
FastAPI service and displays the JSON it gets back:

```python
API_URL = os.environ.get("API_URL", "http://localhost:8000")
...
response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
```

**Why not load the model directly in Streamlit** — it would be simpler code. But then you'd
have *two* programs that both know how to turn raw student data into a prediction: the API and
the UI. The moment you fix a bug in one and forget the other, or retrain the model and update
only one of the two `joblib.load()` call sites, they silently disagree — and nothing tells you
they disagree. **One source of truth**: exactly one process (the FastAPI service) is allowed
to touch the model file. Streamlit is just a client, with no more privilege than a curl
request has.

This mirrors a pattern you'll see constantly in real systems: a mobile app, a web frontend,
and an internal admin tool are often three separate clients of one shared backend API — never
three separate copies of the business logic.

## SIMPLE EXAMPLE — run it standalone (two terminals)

```bash
# terminal 1
uvicorn app.main:app --port 8000

# terminal 2
streamlit run streamlit_app.py
```

Streamlit opens `http://localhost:8501` in your browser automatically. Fill in the form,
click **Predict final score** — under the hood that's the exact same POST request the `curl`
example in the main README sends, just from a form instead of a terminal.

## CODE — reading the important parts

```python
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success("API is up, model loaded")
except requests.exceptions.RequestException as e:
    st.error(f"Cannot reach API at {API_URL}")
```

The sidebar health check exists so a broken/unreachable API fails *visibly*, in the UI itself,
rather than the predict button silently doing nothing. This is the same instinct as the
FastAPI service's own `/health` endpoint — always give the next layer a fast, honest way to
check "is the thing I depend on actually working?"

```python
"enrollment_date": enrollment_date.isoformat(),
```

Streamlit's `date_input` returns a Python `date` object; the API's Pydantic schema expects a
string. `.isoformat()` is the conversion — a small but real example of the UI layer's job
being to translate between "what's convenient for a human filling a form" and "what the API
contract requires."

## INDUSTRY EXAMPLE

Internal tools at most companies — a support team's "look up this customer's account status"
tool, a risk team's "re-score this loan application" tool — are frequently built exactly this
way: Streamlit (or a similar lightweight framework) as a thin, disposable UI in front of a
real, versioned, tested API that other systems also call.

## COMMON MISTAKES

- **Importing the model into Streamlit "just to be quick."** The moment there are two copies
  of the prediction logic, they *will* eventually disagree.
- **Not handling the API being down.** A `requests.exceptions.RequestException` that isn't
  caught crashes the whole Streamlit page with a stack trace a non-technical user can't act
  on.
- **Forgetting `API_URL` is configurable.** Hardcoding `http://localhost:8000` works on your
  laptop and breaks the instant the API and UI run as separate services — this project avoids
  that from the start with `os.environ.get(...)`.

## PRODUCTION CONNECTION

Inside the combined container (Part 3), both processes run on `localhost` relative to *each
other*, so the default `API_URL` just works with zero configuration. If you ever split them
into two separate containers (a legitimate next step, not covered here), `API_URL` is the one
setting you'd change — nothing else in `streamlit_app.py` would need to.

---

# PART 3 — Bundling both into ONE image

## WHY

You now have two things that each run fine alone. The task said: bundle them into **one**
Docker image, so a single `docker run` produces a complete, usable application — one thing to
build, push, and deploy, instead of coordinating two.

## WHAT

A Docker container is built around the idea of one main process (`CMD`/`ENTRYPOINT`). Running
two independent long-lived services (`uvicorn` and `streamlit`) inside one container means
something has to start both, watch both, and shut both down together. That "something" is
`docker/entrypoint.sh` — a small supervisor script.

## HOW — `docker/entrypoint.sh`, step by step

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
```
Start the API **in the background** (`&`), and remember its process ID.

```bash
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" 2>/dev/null; then
        break
    fi
    sleep 1
done
```
Poll `/health` for up to 30 seconds before starting the UI. Without this, the container would
start Streamlit immediately, and anyone loading the page in the first second or two would see
"API unreachable" even though the API is simply still loading the model — a false alarm, not
a real failure. (This reuses `python -c` with `urllib`, not `curl` — the base image doesn't
have `curl` installed, and adding it just for a health check isn't worth the extra image
size.)

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &
UI_PID=$!
```
Now start the UI, also in the background.

```bash
trap 'kill $API_PID $UI_PID 2>/dev/null' SIGTERM SIGINT
wait -n "$API_PID" "$UI_PID"
kill $API_PID $UI_PID 2>/dev/null || true
```
`docker stop` sends `SIGTERM` to the container's main process — that's *this script*, not the
two background processes directly. Without the `trap`, only this script would receive and
react to the signal, and the two children could be left running until Docker's forced-kill
timeout. `wait -n` waits for **either** process to exit (a crash, not just a clean stop); the
moment one does, the script kills the other too. A container where the API silently died but
Streamlit is still serving a broken UI — `docker ps` showing "running" the whole time — is a
worse failure than both going down together, because nothing signals that anything is wrong.

## `Dockerfile` (project root) — the diff from `docker/Dockerfile.api-only`

```dockerfile
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib \
    fastapi "uvicorn[standard]" pydantic \
    streamlit requests                      # + streamlit, requests

COPY streamlit_app.py .                     # + the UI code
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh                  # entrypoint.sh must be executable

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser                                # run as non-root (basic container hygiene)

EXPOSE 8000 8501                            # BOTH ports now

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["./entrypoint.sh"]              # run the supervisor script, not uvicorn directly
```

`HEALTHCHECK` is Docker's *own* built-in health monitor — separate from the entrypoint
script's one-time startup poll. It runs continuously, every 30 seconds, for the life of the
container, and is what makes `docker ps` show `(healthy)` / `(unhealthy)` next to your
container — genuinely useful when this runs on a server you're not actively watching.

## SIMPLE EXAMPLE — build and run the combined image

```bash
docker build -t exam-score-app:1.0 .
docker run -p 8000:8000 -p 8501:8501 exam-score-app:1.0
```

Both ports now, because both services need publishing. Visit `http://localhost:8000/docs` and
`http://localhost:8501` — both come from the *same* container (`docker ps` shows one entry).

## COMMON MISTAKES

- **Only publishing one port.** `-p 8501:8501` alone gives you a UI that loads but whose every
  prediction request fails, because port 8000 was never forwarded.
- **Forgetting `chmod +x entrypoint.sh`.** Without it, the container fails immediately with
  `permission denied`, before either service starts.
- **Skipping the health-poll wait.** Remove that loop and start Streamlit immediately, and
  every fresh container start shows a scary "API unreachable" error for the first couple of
  seconds — technically transient, but looks broken.

## PRODUCTION CONNECTION

Real production systems more often split this into two separate, independently-scalable
containers behind a reverse proxy (nginx, or a cloud load balancer) rather than one bundled
image — you can scale the API to five replicas under heavy prediction traffic without also
running five copies of the UI. Bundling them here is a deliberate teaching simplification: one
image, one `docker run`, one thing to push to Docker Hub and deploy to one EC2 instance. Once
you're comfortable with this, "why would you split them, and how" is a natural next question
to explore on your own.

---

# PART 4 — Docker Hub

## WHY

`docker run exam-score-app:1.0` only works on *this* machine, because the image only exists
in this machine's local Docker image cache. To run it on an EC2 instance, that instance needs
a way to get a copy of the image — without you manually copying gigabytes of files over SSH.

## WHAT

**Docker Hub** is a public **registry** — a hosted store of images, the same relationship
GitHub has to git repositories. `docker push` uploads your image there; `docker pull`
downloads it, from any machine with Docker and a network connection.

## HOW

**1. Create a free account** at hub.docker.com if you don't have one, and note your username.

**2. Log in from your terminal:**
```bash
docker login
```

**3. Tag the image with your Docker Hub username** — Docker Hub identifies images by
`<username>/<repo-name>:<tag>`, so a locally-built `exam-score-app:1.0` needs re-tagging
before it can be pushed:
```bash
docker tag exam-score-app:1.0 <your-dockerhub-username>/exam-score-app:1.0
```

**4. Push it:**
```bash
docker push <your-dockerhub-username>/exam-score-app:1.0
```

**5. Verify** by pulling it back down under a different local name — proving the *registry*
copy works, not just your local build cache:
```bash
docker pull <your-dockerhub-username>/exam-score-app:1.0
```

Or just visit `hub.docker.com/r/<your-dockerhub-username>/exam-score-app` in a browser.

## INDUSTRY EXAMPLE

Most companies don't use public Docker Hub for production images — they use a **private**
registry (AWS ECR, Google Artifact Registry, a self-hosted registry) so proprietary code isn't
publicly downloadable. The workflow (`build → tag → push → pull`) is identical either way;
only the registry's URL and auth method change. Docker Hub here is the free, zero-setup choice
appropriate for a teaching project — not a statement that it's what you'd use at a company.

## COMMON MISTAKES

- **Forgetting to re-tag before pushing.** `docker push exam-score-app:1.0` (without your
  username prefix) fails, or worse, silently tries to push to a Docker Hub *official* image
  namespace you don't own.
- **Pushing a stale image.** If you edit code after building, `docker push` re-uploads
  whatever was last *built*, not your latest source — rebuild first.
- **Treating `:1.0` as fixed.** Nothing stops you from reusing a tag, but for anything beyond
  a class demo, bump the tag (`:1.1`, `:2.0`) on real changes — `latest` silently means
  "whatever was pushed most recently," which is a bad thing to have running unmonitored on a
  server.

## PRODUCTION CONNECTION

This is the exact hand-off point between "developer's laptop" and "everything else." Whether
the next puller is a teammate, a CI runner, or (Part 5) an EC2 instance, they all do the same
`docker pull` — Docker Hub is what makes the image portable beyond the machine that built it.

---

# PART 5 — Running it on AWS EC2

## WHY

Docker Hub makes the image *available* anywhere — but it still needs a machine that's
actually running, with a public IP address, to pull and run it continuously. Your laptop
could technically do this, but it isn't on 24/7, doesn't have a stable public IP, and you
presumably want to close the lid at some point.

## WHAT

**EC2** (Elastic Compute Cloud) rents you a virtual machine on AWS's infrastructure — a real
Linux box, with its own public IP, that you fully control over SSH, billed by the hour (Free
Tier covers a small one for a year at no cost).

The full, step-by-step version of everything below — including screenshots-in-words for the
AWS Console and a section on what to check when a step doesn't work — lives in
**`deploy/aws/EC2_WALKTHROUGH.md`**. Do that walkthrough manually at least once. What follows
here is the summary; the automated path is `deploy/aws/ec2-user-data.sh`.

## HOW

1. **Launch** an Ubuntu 22.04, `t2.micro` EC2 instance.
2. **Open ports 22 (SSH), 8000 (API), and 8501 (Streamlit)** in its security group — AWS's
   firewall. This is the step most likely to trip you up; a container that's running
   perfectly is still unreachable if this is wrong.
3. **SSH in**, install Docker, `docker pull <username>/exam-score-app:1.0`,
   `docker run -d --restart unless-stopped -p 8000:8000 -p 8501:8501 <username>/exam-score-app:1.0`.
4. **Visit `http://<instance-public-ip>:8501`** from your own laptop's browser — not from
   inside the SSH session — to prove it's actually reachable from the outside world.

Or, skip steps 2–3 by pasting `deploy/aws/ec2-user-data.sh` into the **User data** field at
launch time (step 1) — AWS runs it automatically as `root` on first boot, and the instance
comes up already running the container.

## SIMPLE EXAMPLE

```bash
ssh -i exam-score-key.pem ubuntu@<public-ip>
docker pull yourname/exam-score-app:1.0
docker run -d --name exam-score-app --restart unless-stopped -p 8000:8000 -p 8501:8501 yourname/exam-score-app:1.0
```

Then, from your laptop's browser: `http://<public-ip>:8501`.

## INDUSTRY EXAMPLE

A single EC2 instance running one Docker container, as done here, is a legitimate way to run
a small internal tool or a demo — plenty of real internal company tools run exactly this way.
It stops being appropriate once you need: automatic scaling under load, zero-downtime
deploys, or resilience to a single instance dying. That's the point at which teams move to
something like ECS/Fargate or Kubernetes — genuinely more machinery, justified only once a
single box's limitations start to actually hurt.

## COMMON MISTAKES

- **Security group ports not opened** — by far the most common failure, and it looks like a
  Docker problem when it isn't one.
- **Using the private IP** instead of the public IP from outside AWS's network.
- **Forgetting `--restart unless-stopped`** — an instance reboot (routine AWS maintenance, or
  you restarting it) leaves the container not running, with no crash message anywhere obvious.
- **Never terminating the instance** — Free Tier hours run out, and a forgotten always-on demo
  box is a surprisingly common source of unexpected cost outside the Free Tier window.

## PRODUCTION CONNECTION — the whole path, end to end

```
Trained pipeline (.joblib)
        |
        v
FastAPI service  <-- one source of truth for predictions
        |
        v
Streamlit UI  ---- (HTTP only, no model logic of its own)
        |
        v
docker build  -->  ONE image, both processes, one entrypoint.sh
        |
        v
docker push  -->  Docker Hub (the registry)
        |
        v
EC2 instance: docker pull + docker run --restart unless-stopped
        |
        v
A real URL, reachable from any browser, running independently of your laptop
```

Every arrow in that diagram is a real, separate step you can test in isolation — which is
exactly why this guide built them in this order instead of jumping straight to "deploy it."
When something breaks in a real deployment, being able to mentally walk this chain and ask
"which link actually failed?" is the debugging skill this whole exercise is teaching.
