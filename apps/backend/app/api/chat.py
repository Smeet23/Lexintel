"""Chat & RAG API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.services.rag import rag_service
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatConversationCreate,
    ChatConversationResponse,
    ChatConversationDetailResponse,
)
from shared.models import ChatConversation, ChatMessage
from uuid import uuid4
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/conversations", response_model=ChatConversationResponse)
async def create_conversation(
    req: ChatConversationCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new chat conversation for a case

    Args:
        req: ChatConversationCreate with case_id and optional title
        db: AsyncSession for database access

    Returns:
        Created conversation with metadata
    """
    try:
        logger.info(f"[chat] Creating conversation for case: {req.case_id}")

        conversation = ChatConversation(
            id=str(uuid4()),
            case_id=req.case_id,
            title=req.title,
        )

        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        logger.info(f"[chat] Conversation created: {conversation.id}")

        return ChatConversationResponse(
            id=conversation.id,
            case_id=conversation.case_id,
            title=conversation.title,
            token_count=conversation.token_count,
            message_count=conversation.message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    except Exception as e:
        logger.error(f"[chat] Failed to create conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}",
        )


@router.get("/conversations/{conversation_id}", response_model=ChatConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get conversation details with all messages

    Args:
        conversation_id: ID of the conversation
        db: AsyncSession for database access

    Returns:
        Conversation with full message history
    """
    try:
        logger.info(f"[chat] Fetching conversation: {conversation_id}")

        result = await db.execute(
            select(ChatConversation).where(ChatConversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        messages = [
            ChatMessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=msg.content,
                tokens_used=msg.tokens_used,
                source_document_ids=msg.source_document_ids or [],
                created_at=msg.created_at,
            )
            for msg in conversation.messages
        ]

        return ChatConversationDetailResponse(
            id=conversation.id,
            case_id=conversation.case_id,
            title=conversation.title,
            token_count=conversation.token_count,
            message_count=conversation.message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[chat] Failed to fetch conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conversation: {str(e)}",
        )


@router.post("/conversations/{conversation_id}/messages/stream")
async def chat_stream(
    conversation_id: str,
    req: ChatMessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream RAG response

    Performs RAG (Retrieval-Augmented Generation):
    1. Retrieves relevant document chunks
    2. Builds context-aware prompt
    3. Streams response from LLM
    4. Saves conversation history

    Args:
        conversation_id: ID of the conversation
        req: ChatMessageCreate with user message
        db: AsyncSession for database access

    Returns:
        Server-sent events stream of response tokens
    """

    async def generate():
        try:
            logger.info(f"[chat] Processing message in conversation: {conversation_id}")

            # Validate conversation exists
            result = await db.execute(
                select(ChatConversation).where(ChatConversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                yield json.dumps({
                    "error": f"Conversation {conversation_id} not found"
                }).encode()
                return

            # Save user message
            user_message = ChatMessage(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=req.content,
                tokens_used=rag_service.count_tokens(req.content),
            )
            db.add(user_message)
            await db.commit()

            # Search for relevant context
            try:
                context_chunks = await rag_service.search_context(
                    session=db,
                    query=req.content,
                    case_id=conversation.case_id,
                    top_k=5,
                    threshold=0.5,
                )
                logger.info(f"[chat] Retrieved {len(context_chunks)} context chunks")
            except Exception as e:
                logger.warning(f"[chat] Failed to retrieve context: {e}")
                context_chunks = []

            # Build prompt
            prompt = rag_service.build_system_prompt(
                query=req.content,
                context_chunks=context_chunks,
            )

            # Stream response
            full_response = ""
            async for token in rag_service.stream_response(prompt):
                full_response += token
                yield json.dumps({
                    "type": "token",
                    "data": token,
                }).encode() + b"\n"

            # Save assistant message
            assistant_message = ChatMessage(
                id=str(uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                tokens_used=rag_service.count_tokens(full_response),
                source_document_ids=[
                    chunk.get("document_id")
                    for chunk in context_chunks
                    if chunk.get("document_id")
                ],
            )
            db.add(assistant_message)

            # Update conversation counters
            conversation.message_count += 2
            conversation.token_count += (
                rag_service.count_tokens(req.content)
                + rag_service.count_tokens(full_response)
            )

            await db.commit()

            logger.info(f"[chat] Message exchange complete for conversation: {conversation_id}")
            yield json.dumps({
                "type": "done",
                "data": {
                    "tokens_used": assistant_message.tokens_used,
                    "source_documents": len(context_chunks),
                }
            }).encode() + b"\n"

        except Exception as e:
            logger.error(f"[chat] Stream error: {e}")
            yield json.dumps({
                "type": "error",
                "data": str(e),
            }).encode() + b"\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@router.post("/conversations/{conversation_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    conversation_id: str,
    req: ChatMessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and get RAG response (non-streaming)

    Args:
        conversation_id: ID of the conversation
        req: ChatMessageCreate with user message
        db: AsyncSession for database access

    Returns:
        Assistant's response message
    """
    try:
        logger.info(f"[chat] Processing message in conversation: {conversation_id}")

        # Validate conversation exists
        result = await db.execute(
            select(ChatConversation).where(ChatConversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        # Save user message
        user_message = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=req.content,
            tokens_used=rag_service.count_tokens(req.content),
        )
        db.add(user_message)
        await db.commit()

        # Search for relevant context
        try:
            context_chunks = await rag_service.search_context(
                session=db,
                query=req.content,
                case_id=conversation.case_id,
                top_k=5,
                threshold=0.5,
            )
            logger.info(f"[chat] Retrieved {len(context_chunks)} context chunks")
        except Exception as e:
            logger.warning(f"[chat] Failed to retrieve context: {e}")
            context_chunks = []

        # Build prompt
        prompt = rag_service.build_system_prompt(
            query=req.content,
            context_chunks=context_chunks,
        )

        # Get response from LLM
        full_response = ""
        async for token in rag_service.stream_response(prompt):
            full_response += token

        # Save assistant message
        assistant_message = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
            tokens_used=rag_service.count_tokens(full_response),
            source_document_ids=[
                chunk.get("document_id")
                for chunk in context_chunks
                if chunk.get("document_id")
            ],
        )
        db.add(assistant_message)

        # Update conversation counters
        conversation.message_count += 2
        conversation.token_count += (
            rag_service.count_tokens(req.content)
            + rag_service.count_tokens(full_response)
        )

        await db.commit()
        await db.refresh(assistant_message)

        logger.info(f"[chat] Message exchange complete for conversation: {conversation_id}")

        return ChatMessageResponse(
            id=assistant_message.id,
            conversation_id=assistant_message.conversation_id,
            role=assistant_message.role,
            content=assistant_message.content,
            tokens_used=assistant_message.tokens_used,
            source_document_ids=assistant_message.source_document_ids or [],
            created_at=assistant_message.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[chat] Failed to process message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )
