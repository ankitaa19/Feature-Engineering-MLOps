# Dockerfile
#
# TEACHING STEP 2/3's destination: ONE image containing BOTH the FastAPI prediction
# service AND the Streamlit UI, so `docker run` on this image starts a complete,
# usable application -- not just an API a developer has to curl.
#
# This evolved directly from docker/Dockerfile.api-only (build and run that one first --
# see the README's Docker section) by adding: Streamlit + requests to the installed
# packages, the streamlit_app.py file, both ports exposed, and docker/entrypoint.sh as
# the command instead of a single `uvicorn ...` -- everything else is unchanged.
#
# IMPORTANT: train the model first (run the notebook, or see README) so
# models/exam_score_pipeline.joblib exists before building -- this Dockerfile copies it
# in; it does not train it.
#
# Build:
#     docker build -t <dockerhub-username>/exam-score-app:1.0 .
# Run:
#     docker run -p 8000:8000 -p 8501:8501 <dockerhub-username>/exam-score-app:1.0
#     -> API:       http://localhost:8000/docs
#     -> Streamlit: http://localhost:8501

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first (better layer caching -- see docker/Dockerfile.api-only's comment
# for why), now including Streamlit + requests for the UI.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib \
    fastapi "uvicorn[standard]" pydantic \
    streamlit requests

# Application code + the trained artifact
COPY features.py .
COPY app/ app/
COPY models/ models/
COPY streamlit_app.py .
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run as a non-root user -- a basic container security practice.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# The API's port and the UI's port -- both need publishing (`-p 8000:8000 -p 8501:8501`)
# for both services to actually be reachable from outside the container.
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
