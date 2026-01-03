from sqlalchemy import Column, String, Text, DateTime, Enum
from app.models.base import Base, TimestampMixin
import enum

class CaseStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"

class Case(Base, TimestampMixin):
    __tablename__ = "cases"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    case_number = Column(String, unique=True, nullable=False, index=True)
    practice_area = Column(String, nullable=False)
    status = Column(Enum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False, index=True)
    description = Column(Text)

    # Relationships
    # documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    # chats = relationship("ChatConversation", back_populates="case", cascade="all, delete-orphan")
