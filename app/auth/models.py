from sqlmodel import SQLModel, Field
from pydantic import EmailStr
from typing import Optional
from datetime import datetime, timezone


class UserBase(SQLModel):
    email: EmailStr = Field(unique=True)
    first_name: str
    last_name: str
    title: str


class User(UserBase, table=True):
    __tablename__ = "user" # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": "CURRENT_TIMESTAMP"},
    )

class UserPublic(UserBase):
    id:int