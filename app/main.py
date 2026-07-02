from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.auth.routes import router as auth_router
from app.location.routes import router as location_router

app = FastAPI()


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
async def validation_exception_handler(request: Request, exc: RequestValidationError,):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message":"Invalid request data",
            "data":exc.errors()
        }
    )

app.include_router(auth_router)
app.include_router(location_router)


@app.get("/")
def root():
    return {"Hello": "World"}
