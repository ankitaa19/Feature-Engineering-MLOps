#!/bin/bash
# docker/entrypoint.sh
#
# TEACHING STEP 3 of 3: runs BOTH the FastAPI service and the Streamlit UI inside ONE
# container. Docker containers are built around running one main process, so bundling two
# real services needs a small supervisor script like this one -- deliberately a plain bash
# script rather than pulling in a process manager like supervisord, since a few lines of
# bash is easier for a student to actually read and understand line by line.
#
# What it does, in order:
#   1. Starts the FastAPI service in the background.
#   2. Waits until it reports healthy (so the UI's sidebar doesn't open showing
#      "API unreachable" for the first few seconds of every container start).
#   3. Starts the Streamlit UI in the background too.
#   4. Waits for EITHER process to exit, then stops the other and exits with the same
#      code -- so a crashed API doesn't leave a container that LOOKS alive (`docker ps`
#      shows it running) while silently serving a broken UI, and `docker stop` (which
#      sends SIGTERM) actually stops both processes instead of leaving one behind.

set -e

echo "[entrypoint] starting FastAPI on :8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "[entrypoint] waiting for the API to report healthy..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" 2>/dev/null; then
        echo "[entrypoint] API is healthy."
        break
    fi
    sleep 1
done

echo "[entrypoint] starting Streamlit on :8501 ..."
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &
UI_PID=$!

# Forward SIGTERM/SIGINT (what `docker stop` sends) to BOTH child processes -- without
# this, only this script's own process gets the signal and the children can be left
# running until Docker's forced-kill timeout.
trap 'echo "[entrypoint] stopping..."; kill $API_PID $UI_PID 2>/dev/null' SIGTERM SIGINT

# If either process exits on its own (e.g. a crash), stop the other one too and exit --
# a container with only half its services alive is a bug, not a degraded-but-ok state.
wait -n "$API_PID" "$UI_PID"
EXIT_CODE=$?
echo "[entrypoint] one process exited (code $EXIT_CODE) -- stopping the other."
kill $API_PID $UI_PID 2>/dev/null || true
exit $EXIT_CODE
