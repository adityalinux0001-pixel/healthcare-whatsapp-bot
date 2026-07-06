"""
STEP 4 (dev mode only): local development entrypoint.

This runs a single uvicorn worker with autoreload — convenient while
coding, but it does NOT exercise the multi-worker behavior this project
was just migrated to support (steps 1-3 only actually matter once you
have more than one worker process). For anything resembling production,
or to actually test that the Redis-backed semaphore/idempotency guard
work correctly under multiple workers, use `start.sh` / the gunicorn
command below instead:

    gunicorn app.main:app \\
        --worker-class uvicorn.workers.UvicornWorker \\
        --workers ${WEB_CONCURRENCY:-4} \\
        --bind 0.0.0.0:8000 \\
        --timeout 120

See start.sh for the version with env-driven worker count and logging
flags wired up, and docker-compose.yml for how it's launched in the
compose stack.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
