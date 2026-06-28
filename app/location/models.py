from sqlmodel import SQLModel, Field
from typing import Optional


class Location(SQLModel, table=True):
    __tablename__ = "location" # type: ignore
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")