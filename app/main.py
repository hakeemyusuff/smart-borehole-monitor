from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.scheduler import start_scheduler, stop_scheduler

from app.auth.routes import router as auth_router
from app.location.routes import router as location_router
from app.borehole.routes import router as borehole_router
from app.sensor.routes import router as sensor_router
from app.weather.routes import router as weather_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Booting the weather synchronization engine...")
    start_scheduler()
    yield
    print("Shutting down: Safely closing scheduler threads...")
    stop_scheduler()


app = FastAPI(
    title="BoreSense",
    description="IoT-based real-time groundwater level monitoring and pump scheduling system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "data": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Invalid request data",
            "data": exc.errors(),
        },
    )


app.include_router(auth_router, prefix="/api")
app.include_router(location_router, prefix="/api")
app.include_router(borehole_router, prefix="/api")
app.include_router(sensor_router, prefix="/api")
app.include_router(weather_router, prefix="/api")


@app.get("/")
def health():
    return {"status": "ok"}
