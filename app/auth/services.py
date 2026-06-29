from pwdlib import PasswordHash
from pydantic import EmailStr
from app.core.config import settings
from typing import Any
from datetime import datetime, timezone, timedelta
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from app.auth.models import User
import jwt

password_hash = PasswordHash.recommended()


# Function for hashing password
def hash_password(plain_password: str) -> str:
    return password_hash.hash(plain_password)


# function for verifying password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


# Function for creating an access token for user login
def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=24)
    }
    access_token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return access_token

# Function for decoding the token to makes sure it matches when the user makes a request
def decode_access_token(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        user_id = int(payload["sub"])
        return user_id
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token") 


async def register_user(
    email: EmailStr,
    title: str,
    first_name: str,
    last_name: str,
    password: str,
    session: AsyncSession,
) -> User:
    result = await session.exec(select(User).where(User.email == email))
    existing = result.first()
    if existing:
        raise ValueError("Email is already registered")
    
    user = User(
        email=email,
        title=title,
        first_name=first_name,
        last_name=last_name,
        hashed_password=hash_password(password),
    )
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def login_user(
    email: EmailStr,
    password: str,
    session: AsyncSession,
) -> str:
    result = await session.exec(select(User).where(User.email == email))
    user = result.first()

    if user is None or not verify_password(password, user.hashed_password):
        raise ValueError("Invalid email or password")

    if user.id is None:
        raise ValueError("User does not have an ID")
    
    return create_access_token(user.id)
