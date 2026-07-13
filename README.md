# BoreSense

An IoT-based real-time groundwater level monitoring and machine learning-driven pump scheduling system for sustainable borehole management.

BoreSense collects continuous water-level and flow data from boreholes via ESP32-connected sensors, enriches it with meteorological data, and is designed to support recharge-rate prediction and intelligent pump scheduling. This repository contains the **backend API**.

> Status: active development. Pump scheduling and the machine-learning prediction layer are planned and not yet implemented in this backend.

---

## Tech Stack

- **Framework:** FastAPI
- **ORM / models:** SQLModel (built on SQLAlchemy + Pydantic)
- **Database:** PostgreSQL (run locally via Docker Compose)
- **Migrations:** Alembic
- **Package manager:** uv
- **Background scheduling:** APScheduler (3.11.3)
- **HTTP client:** httpx (for the Open-Meteo weather API)
- **Password / device-key hashing:** pwdlib (Argon2)
- **Auth tokens:** PyJWT (HS256)
- **Weather data:** Open-Meteo API (no API key required)

**Python version:** 3.13+

---

## Architecture

The project uses a **feature-based structure** — each domain area owns its own models, routes, and services rather than grouping by file type.

```
app/
├── main.py            # FastAPI app entry point, routers, lifespan
├── core/
│   ├── config.py      # settings loaded from .env (pydantic-settings)
│   ├── database.py    # async engine, session maker, get_session dependency
│   ├── schemas.py     # shared schemas (e.g. ApiResponse envelope)
│   └── scheduler.py   # APScheduler setup for background weather fetching
├── auth/              # user registration, login, JWT, dependencies
├── location/          # locations (with latitude/longitude)
├── borehole/          # boreholes
├── sensor/            # sensors, device auth, reading ingestion & listing
├── weather/           # Open-Meteo integration, weather storage, tasks
├── pump/              # pump models (endpoints planned)
├── ml/                # prediction models (planned)
└── background/        # background task wiring
```

### Key design choices

- **Service / route separation:** business logic lives in `services.py`; routes handle HTTP only. Services raise plain `ValueError` on failure; routes translate these into HTTP status codes.
- **Consistent response envelope:** all endpoints return an `ApiResponse` wrapper with `status`, `message`, and `data`.
- **Two authentication paths:**
  - **User auth (JWT):** for all user-facing resource endpoints. Send `Authorization: Bearer <token>`.
  - **Device auth (device key):** for sensor reading ingestion. The ESP32 sends its own sensor id and raw device key in request headers, verified against a stored hash.
- **Ownership scoping:** user-facing queries are scoped so a user only accesses their own locations, boreholes, sensors, and readings.

---

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) installed
- Docker and Docker Compose (for the local PostgreSQL instance)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/hakeemyusuff
cd smart-borehole-monitor
uv sync
```

> Note: the internal repository/package name is `smart-borehole-monitor`; "BoreSense" is the product display name.

### 2. Environment variables

Copy `.env.example` to `.env` and fill in values:

```
DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/<db_name>
SECRET_KEY=<a long random string>
DEBUG=true
DB_NAME=<db_name>
DB_USER=<user>
DB_PASSWORD=<password>
ALLOWED_ORIGINS=http://localhost:5173
ENABLE_SCHEDULER=true
```

- `DATABASE_URL` — async PostgreSQL connection string used by the app.
- `SECRET_KEY` — used to sign JWT access tokens. Generate a strong random value (for example, using Python's `secrets` module).
- `DEBUG` — enables verbose SQL echoing and debug-level logs when `true`.
- `DB_NAME`, `DB_USER`, `DB_PASSWORD` — used by Docker Compose to provision the PostgreSQL container.
- `ALLOWED_ORIGINS` — comma-separated list of frontend origins allowed by CORS. In production, set this to your deployed frontend URL(s).
- `ENABLE_SCHEDULER` — set to `false` on all but one app process when running multiple workers (see [Deployment](#deployment)).

### 3. Start the database

```bash
docker compose up -d
```

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, and interactive docs at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

All endpoints return the standard `ApiResponse` envelope:

```json
{
  "status": "success",
  "message": "",
  "data": {}
}
```

### Auth

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/auth/register` | Register a new user | None |
| POST | `/auth/login` | Log in and receive a JWT access token | None |

### Locations

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/locations/` | Create a location (with latitude/longitude) | JWT |
| GET | `/locations/` | List the current user's locations | JWT |
| GET | `/locations/{location_id}` | Retrieve a single location | JWT |

### Boreholes

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/boreholes/` | Create a borehole | JWT |
| GET | `/boreholes/` | List the current user's boreholes | JWT |
| GET | `/boreholes/{borehole_id}` | Retrieve a single borehole | JWT |

### Sensors

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/sensors/` | Register a sensor. Registering an ESP32 returns a one-time device key | JWT |
| GET | `/sensors/boreholes/{borehole_id}` | List sensors for a borehole | JWT |
| GET | `/sensors/{sensor_id}` | Retrieve a single sensor | JWT |

### Readings (ingestion + listing)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/sensors/readings/water-level` | Ingest a water-level reading | Device key |
| GET | `/sensors/readings/water-level/{borehole_id}/{sensor_id}` | List water-level readings | JWT |
| POST | `/sensors/readings/flow-reading` | Ingest a flow reading | Device key |
| GET | `/sensors/readings/flow-reading/{borehole_id}/{sensor_id}` | List flow readings | JWT |

**Device-authenticated ingestion** requires these headers:

- `X-Device-Id` — the ESP32's own sensor id
- `X-Device-Key` — the raw device key issued when the ESP32 was registered

The request body contains the reading-producing sensor id and the reading value.

### Weather

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/weathers/fetch/{location_id}` | Manually fetch and store current weather for a location | JWT |
| GET | `/weathers/{location_id}` | List stored weather records for a location | JWT |

Weather is also fetched automatically on a schedule (see below).

---

## Background Weather Fetching

A background scheduler (APScheduler) periodically fetches current weather from the Open-Meteo API for every location that has coordinates, and stores it. The scheduler is started and stopped via the FastAPI application lifespan, and can be toggled per-process with `ENABLE_SCHEDULER`.

> When running with multiple worker processes, only ONE worker should have `ENABLE_SCHEDULER=true` — otherwise every worker fetches weather on the same interval. See [Deployment](#deployment).

---

## Deployment

### 1. Environment

Set the same variables as in [Setup — Environment variables](#2-environment-variables) on the target platform's secret store. In particular:

- Set `SECRET_KEY` to a fresh, strong value — do not reuse the dev secret.
- Set `DEBUG=false`.
- Set `ALLOWED_ORIGINS` to your deployed frontend origin(s), comma-separated.
- Set `ENABLE_SCHEDULER=true` on exactly one process; `false` everywhere else.

### 2. Run migrations before starting the app

On every release, run:

```bash
alembic upgrade head
```

This must complete successfully before the app process starts serving traffic. On platforms with a "release" or "pre-deploy" hook (Render, Railway, Fly, Heroku), put it there. Otherwise, run it manually or from a start script before `uvicorn`.

### 3. Start command

Run under a production ASGI server. Because of the scheduler note above, pin to a single worker unless you split the scheduler out:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers
```

If you need more workers, run one instance with `ENABLE_SCHEDULER=true --workers 1` and scale the rest with `ENABLE_SCHEDULER=false`.

---

## Data Flow (high level)

1. A user registers and logs in.
2. The user creates a **location** (with coordinates) and one or more **boreholes** under it.
3. **Sensors** are registered on a borehole — typically an ESP32 (which receives a one-time device key), a pressure transducer, and a flow meter.
4. The ESP32 reads its attached sensors and sends readings to the ingestion endpoints, authenticating with its device key.
5. The backend stores readings and updates sensor heartbeat status.
6. Weather data is fetched for each location, both on demand and on a schedule.

---

## Development Notes

- Generated Alembic migration files should be reviewed before running `alembic upgrade head`.
- All timestamps are stored timezone-aware (UTC).
- Planned/next: pump control and scheduling endpoints, and the machine-learning recharge-prediction layer.

---

## License

<FILL IN — no license is currently specified for this project.>