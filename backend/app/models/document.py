from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, ARRAY, Enum
from sqlalchemy.orm import relationship
import enum
from app.models.base import Base, TimestampMixin

class DocumentType(str, enum.Enum):
    BRIEF = "brief"
    COMPLAINT = "complaint"
    DISCOVERY = "discovery"
    STATUTE = "statute"
    TRANSCRIPT = "transcript"
    CONTRACT = "contract"
    EVIDENCE = "evidence"
    OTHER = "other"

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    INDEXED = "indexed"
    FAILED = "failed"

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    type = Column(Enum(DocumentType), nullable=False)
    extracted_text = Column(Text)
    page_count = Column(Integer)
    file_size = Column(Integer)
    file_path = Column(String)  # Local file path
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    error_message = Column(Text)
    indexed_at = Column(DateTime)

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(String)  # pgvector stored as string
    search_vector = Column(String)  # PostgreSQL tsvector

    # Relationships
    document = relationship("Document", back_populates="chunks")

class ChatConversation(Base, TimestampMixin):
    __tablename__ = "chat_conversations"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, default="Untitled Conversation")
    token_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)

    # Relationships
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("chat_conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    source_document_ids = Column(ARRAY(String), default=[])

    # Relationships
    conversation = relationship("ChatConversation", back_populates="messages")
