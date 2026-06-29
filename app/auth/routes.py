from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import get_session
from app.auth.models import UserPublic
from app.auth.services import register_user, login_user
from pydantic import BaseModel, EmailStr
from app.core.schemas import ApiResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    title: str
    first_name: str
    last_name: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post(
    "/register",
    response_model=ApiResponse[UserPublic],
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        user = await register_user(**payload.model_dump(), session=session)
        return ApiResponse(
            status="success",
            message="User registered successfully",
            data=user,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=ApiResponse[TokenResponse])
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
):
    try:
        token = await login_user(
            email=form_data.username,
            password=form_data.password,
            session=session,
        )
        return ApiResponse(
            status="success",
            message="Login Successful",
            data=TokenResponse(access_token=token),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
