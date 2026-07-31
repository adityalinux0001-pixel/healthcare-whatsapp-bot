# whatsapp-bot-multiworker

A multi-worker WhatsApp bot built with Python. This repository provides a scalable worker-based architecture for handling WhatsApp messages, audio ingestion, LLM-based processing, and background tasks.

## Features

- Scalable worker architecture (Kafka, RQ, Celery, Huey)
- Context-aware conversations backed by PostgreSQL
- Audio transcription and processing
- LLM integration
- Redis-backed queueing and idempotency helpers
- Docker and docker-compose support for easy deployment

## Quick Start (local)

Prerequisites:
- Python 3.9+ (3.11 recommended)
- pip
- Redis (or use Docker Compose)

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy or create your environment file

Place configuration values in `.env` at the repository root. Example variables used by the app include WhatsApp keys, Redis URL, and PhonePe credentials. (See `app/config.py`.)

3. Run the app locally

```powershell
# run the main web service or entrypoint
python run.py

# start a worker process
python -m app.workers.worker
```

### Setup and start (detailed)

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill values:

```powershell
copy .env.example .env
# then edit .env with your credentials
notepad .env
```

4. Start services (in separate terminals):

Terminal A (main web service):

```powershell
python run.py
```

Terminal B (worker process):

```powershell
python -m app.workers.worker
```

If you prefer Docker Compose, see the Quick Start (Docker) section above.

## Quick Start (Docker)

Start the full stack with Docker Compose:

```powershell
docker-compose up --build
```

This will build the image(s) and run services as configured in `docker-compose.yml`.

## Project layout

- `run.py` — project entry for the main service (local dev)
- `start.sh` — convenience script for Unix-like systems and Docker entrypoint
- `app/` — application package
  - `api/` — FastAPI webhook receiver and HTTP entrypoint (`main.py`)
  - `core/` — core infrastructure (config, queues, idempotency, redis, kafka)
  - `models/` — domain models and Pydantic schemas (`schemas.py`)
  - `services/` — business logic and external integrations (LLM, WhatsApp, PhonePe, audio processing, memory, onboarding)
  - `workers/` — background worker processes (inbound, outbound, dead letter)

## Configuration

Put all runtime secrets and settings in `.env` (root). Typical entries:

- `REDIS_URL` — Redis connection string
- `WHATSAPP_API_TOKEN` — WhatsApp API token/credentials
- `PHONEPE_CLIENT_ID` / `PHONEPE_CLIENT_SECRET` / `PHONEPE_CLIENT_VERSION` — payment provider credentials
- `PHONEPE_WEBHOOK_USERNAME` / `PHONEPE_WEBHOOK_PASSWORD` — webhook credentials set in the PhonePe dashboard
- `PHONEPE_REDIRECT_URL` — public URL of the `/phonepe/redirect` page
- `OPENAI_API_KEY` (or equivalent) — LLM provider key

Check `app/config.py` for the full list of environment variables the app expects.

## Development notes

- Use multiple worker processes for scaling: run `python worker.py` in separate terminals or orchestrate via Docker Compose.
- Static and audio files are stored under `audio_storage/` by default.
- If you change Python deps, update `requirements.txt` and rebuild the Docker image.

## Troubleshooting

- If you get Redis connection errors, ensure `REDIS_URL` is correct and Redis is running.
- For issues with external APIs, verify credentials in `.env` and consult the logs produced by `run.py`/`worker.py`.

## Contribution

Feel free to open issues or PRs. Keep changes focused and include tests where applicable.

## License

This repository does not include a license file. Add one if you plan to publish or share the project publicly.

---

If you want, I can:
- add an example `.env.example` with the required variables
- add a `Makefile` or PowerShell scripts for common tasks
- include detailed architecture diagrams and sequence flows
