from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ChatMessageCreate(BaseModel):
    role: str  # user, assistant
    content: str

class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    tokens_used: int
    source_document_ids: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ChatConversationCreate(BaseModel):
    case_id: str
    title: Optional[str] = "Untitled Conversation"

class ChatConversationResponse(BaseModel):
    id: str
    case_id: str
    title: str
    token_count: int
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatConversationDetailResponse(ChatConversationResponse):
    messages: List[ChatMessageResponse] = []
