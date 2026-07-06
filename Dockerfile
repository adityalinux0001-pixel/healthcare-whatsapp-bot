FROM python:3.11-slim

WORKDIR /app

# System deps: libpq for psycopg (even the [binary] extra pulls in some
# runtime shared libs on slim images), and build tools kept minimal since
# psycopg[binary] ships prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# audio_storage is where voice notes get written — mount this as a
# volume in compose so files survive container restarts/recreation.
RUN mkdir -p audio_storage

EXPOSE 8000

# Default to the multi-worker production entrypoint. Override the
# command in docker-compose.yml (or `docker run`) to `python run.py` for
# a single-process autoreloading dev loop instead.
CMD ["./start.sh"]
