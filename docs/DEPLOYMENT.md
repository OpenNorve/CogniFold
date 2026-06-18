<!--
================================================================================
README snippet (for lead to insert)
--------------------------------------------------------------------------------
The block below is a ready-to-paste Quickstart for the README. Copy it into
README.md wherever appropriate; it is duplicated here only as a handoff.
================================================================================

## Quickstart (run the API)

```bash
make dev                      # install package + dev dependencies
cp .env.example .env          # then add your GOOGLE_API_KEY
make serve                    # start the API at http://127.0.0.1:8000
```

Open http://127.0.0.1:8000/docs for the interactive API docs, or check
http://127.0.0.1:8000/health. Prefer containers? `docker compose up`.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker and Cloud Run.

================================================================================
End README snippet.
================================================================================
-->

# Deployment

CogniFold ships a FastAPI service (`cognifold.service.wsgi:app`) served by
uvicorn. This guide covers three tiers:

1. [Local](#1-local) — run it on your machine with `make serve`.
2. [Docker](#2-docker) — run it in a container with `docker compose up`.
3. [Cloud Run](#3-google-cloud-run) — the existing, already-wired CD pipeline.

All three read the same `COGNIFOLD_*` environment variables; see the
[Configuration reference](#configuration-reference) below.

---

## 1. Local

### Prerequisites

- Python 3.9+ (the project targets 3.11 in CI).
- A `GOOGLE_API_KEY` for the agent (the default model is a Gemini model; see
  `config.example.yaml`).

### Steps

```bash
# 1. Install the package + dev dependencies (editable install).
make dev

# 2. Configure secrets and settings.
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (and OPENAI_API_KEY if used).

# 3. Start the service.
make serve
```

`make serve` wraps `scripts/start_server.sh`, which runs
`python -m cognifold serve …` (uvicorn by default). With no overrides it binds
to `127.0.0.1:8000`, uses the `file` session backend, and runs with auth
**disabled** (no API key set).

> Note: `make serve` requires the `service` extra (FastAPI + uvicorn). `make
> dev` installs only the `dev` extra. If you hit a missing-FastAPI error, run
> `pip install -e ".[dev,service,agent]"` (or `".[service,agent,search]"` to
> match the Docker image).

### Verify

```bash
curl http://127.0.0.1:8000/health          # -> 200
curl http://127.0.0.1:8000/api/v1/ready     # readiness (store health)
open http://127.0.0.1:8000/docs             # interactive OpenAPI docs
```

Health is also exposed at `/api/v1/health`.

### Overriding defaults

Every setting is an environment variable. Override inline or via `.env`:

```bash
COGNIFOLD_HOST=0.0.0.0 COGNIFOLD_PORT=9000 make serve
COGNIFOLD_API_KEY=secret123 make serve          # enable API-key auth
COGNIFOLD_WORKERS=4 COGNIFOLD_GUNICORN=1 make serve   # gunicorn, 4 workers
```

See the [Configuration reference](#configuration-reference) for the full list.

---

## 2. Docker

The repository includes a `Dockerfile` (`python:3.11-slim`, installs the
`service,agent,search` extras, runs uvicorn on `0.0.0.0:8000`) and a
`docker-compose.yml` that wires it up for local use.

### With docker compose (recommended)

```bash
cp .env.example .env     # configure GOOGLE_API_KEY etc. (env_file for the container)
docker compose up        # build on first run, then start the API
```

The API is available at http://localhost:8000 (docs at `/docs`). The compose
service includes a `/health` healthcheck, so `docker compose ps` shows the
container as `healthy` once it is ready.

You can also run it via the Makefile:

```bash
make serve-docker        # == docker compose up
```

### With raw docker

```bash
docker build -t cognifold:local .
docker run --rm -p 8000:8000 --env-file .env cognifold:local
```

### Switching to the Redis session backend

The default is the `file` session backend, which works out of the box. To use
Redis instead:

1. Uncomment the `redis` service (and the `depends_on` block) in
   `docker-compose.yml`.
2. In `.env`, set:
   ```bash
   COGNIFOLD_SESSION_BACKEND=redis
   COGNIFOLD_REDIS_URL=redis://redis:6379/0
   ```
   (`redis` is the compose service name, resolvable on the compose network.)
3. `docker compose up`.

---

## 3. Google Cloud Run

Deployment to Cloud Run is **already wired** via
`.github/workflows/cd.yml`. You do not run anything by hand to deploy — you
promote code to the stable branch and the pipeline builds, pushes, and deploys.

### What triggers it

A **push to the `cognifold-stable` branch**. The typical flow is to promote a
tested commit to that branch (a `promote-to-stable` workflow exists for this);
the push then fires the CD pipeline.

### What the pipeline does

1. **Build & push** the Docker image to **two** registries:
   - GHCR: `ghcr.io/<owner>/<repo>`
   - GCP Artifact Registry:
     `<region>-docker.pkg.dev/<project>/cognifold/cognifold`

   Tagged `latest` and `sha-<short-sha>`. GCP auth uses **Workload Identity
   Federation** (no long-lived keys).

2. **Deploy to Cloud Run** (service `cognifold`, `production-gcp`
   environment) with:
   - 2 vCPU / 2Gi memory, `--min-instances=1 --max-instances=5`,
     `--concurrency=4`, `--timeout=300`
   - `--session-affinity`, `--no-cpu-throttling`, `--cpu-boost`, port `8000`
   - a dedicated runtime service account, a VPC connector
     (`cognifold-connector`, egress `private-ranges-only`), and
     `--allow-unauthenticated`
   - `COGNIFOLD_SESSION_BACKEND=redis` pointing at a Memorystore Redis host

3. **Post-deploy health check** against `/health` (and prints `/ready`).

### Required GCP setup (prerequisites)

The workflow expects these to already exist in the GitHub repo/environment:

**Repository / environment variables (`vars`)**
| Name | Purpose |
| --- | --- |
| `GCP_PROJECT_ID` | Target GCP project. |
| `GCP_REGION` | Deploy region (defaults to `us-central1`). |

**Secrets**
| Name | Purpose |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF provider resource name for keyless auth. |
| `GCP_SERVICE_ACCOUNT` | Deployer service account email (impersonated via WIF). |
| `REDIS_HOST` | Memorystore Redis host IP for the session backend. |
| `COGNIFOLD_SUPABASE_KEY` | Supabase key (set as a Cloud Run env var). |

**GCP-side resources / Secret Manager entries**
| Name | Purpose |
| --- | --- |
| Artifact Registry repo `cognifold` | Holds the pushed images. |
| Runtime SA `cognifold-runtime@<project>.iam.gserviceaccount.com` | Cloud Run service identity. |
| VPC connector `cognifold-connector` | Private egress to Memorystore. |
| Secret `google-api-key` | Mounted as `GOOGLE_API_KEY`. |
| Secret `cognifold-api-key` | Mounted as `COGNIFOLD_API_KEY` (enables auth). |

> Cloud Run runs with `COGNIFOLD_API_KEY` set, so the deployed service
> **requires** an API key, unlike the local default.

### Promote flow (summary)

1. Land your change on the main development branch.
2. Promote it to `cognifold-stable` (e.g. via the `promote-to-stable`
   workflow or a fast-forward of that branch).
3. The push to `cognifold-stable` triggers `cd.yml`, which builds, pushes,
   deploys, and health-checks automatically.

---

## Configuration reference

All settings are environment variables (also settable in `.env`). Defaults are
those of `scripts/start_server.sh`.

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | _(none)_ | API key for the Gemini agent. Required for agent runs. |
| `OPENAI_API_KEY` | _(none)_ | API key for OpenAI, if used. |
| `COGNIFOLD_HOST` | `127.0.0.1` | Bind host. Use `0.0.0.0` to expose externally / in containers. |
| `COGNIFOLD_PORT` | `8000` | Bind port. |
| `COGNIFOLD_LOG_LEVEL` | `info` | `debug` \| `info` \| `warning` \| `error`. |
| `COGNIFOLD_WORKERS` | `1` | Worker process count. |
| `COGNIFOLD_GUNICORN` | `0` | Set `1` to use gunicorn instead of uvicorn. |
| `COGNIFOLD_API_KEY` | _(none)_ | Comma-separated API keys. Empty = auth disabled. |
| `COGNIFOLD_PERSIST_DIR` | `./sessions` | Directory for the `file` session backend. |
| `COGNIFOLD_SESSION_BACKEND` | `file` | Session store: `file` or `redis` (Cloud Run uses `redis`). |
| `COGNIFOLD_REDIS_URL` | `redis://localhost:6379/0` | Redis URL when backend is `redis`. |
| `COGNIFOLD_MAX_SESSIONS` | `100` | Max concurrent sessions. |
| `COGNIFOLD_SESSION_TTL_HOURS` | `24` | Session time-to-live. |
| `COGNIFOLD_SUPABASE_URL` | _(none)_ | Supabase URL for persistent storage (optional). |
| `COGNIFOLD_SUPABASE_KEY` | _(none)_ | Supabase key (optional). |
| `COGNIFOLD_ENABLE_GRAPH_SYNC` | `false` | Enable graph sync to Supabase (optional). |

### Session backends

- **`file`** (default) — sessions persisted under `COGNIFOLD_PERSIST_DIR`.
  Simplest; good for local and single-instance.
- **`redis`** — sessions in Redis (`COGNIFOLD_REDIS_URL`). Required for
  multi-instance / Cloud Run, where state must be shared across instances.
- **Supabase** — optional persistent storage layer configured via the
  `COGNIFOLD_SUPABASE_*` variables; install the `supabase` extra.

Optional dependency extras (from `pyproject.toml`): `service`, `agent`,
`search`, `redis`, `supabase`, `dev`, `production`. The Docker image installs
`service,agent,search`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `make serve` fails with a FastAPI/uvicorn import error | The `service` extra isn't installed. Run `pip install -e ".[dev,service,agent]"`. |
| `ModuleNotFoundError: cognifold` | Package not installed. Run `make dev`, or set `PYTHONPATH=src` per `scripts/start_server.sh`. |
| Agent calls fail / empty responses | `GOOGLE_API_KEY` not set in `.env` (or the environment). |
| `/health` ok but `/api/v1/ready` not | Store backend not reachable — for `redis`, check `COGNIFOLD_REDIS_URL` / that Redis is up. |
| `401 Unauthorized` from the API | `COGNIFOLD_API_KEY` is set, so auth is enabled. Send the key, or unset it to disable auth locally. |
| Can't reach the container from the host | Bind to `0.0.0.0` (the Docker image already does) and ensure `-p 8000:8000`. |
| `docker compose up` can't read settings | Missing `.env`. Run `cp .env.example .env` first. |
| Redis backend connection refused in compose | Enable the `redis` service in `docker-compose.yml` and set `COGNIFOLD_REDIS_URL=redis://redis:6379/0`. |
