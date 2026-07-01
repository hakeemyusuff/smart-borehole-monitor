from sqlmodel import SQLModel, Field
from pydantic import EmailStr
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import text, DateTime
from app.core.schemas import timestamp


class UserBase(SQLModel):
    email: EmailStr = Field(unique=True)
    first_name: str
    last_name: str
    title: str


class User(UserBase, table=True):
    __tablename__ = "user"  # type: ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    created_at: datetime = timestamp_field()


class UserPublic(UserBase):
    id: int
