"""Audit logging helper for tracking matter activity"""
import logging
import uuid

logger = logging.getLogger(__name__)

try:
    from backend.models import AuditLog
except ImportError:
    try:
        from models import AuditLog
    except ImportError:
        from ..models import AuditLog


def log_activity(db, matter_id, action, details=None, sources=None, user="System"):
    """
    Log an activity to the audit trail. Fire-and-forget: never raises.

    Args:
        db: SQLAlchemy database session
        matter_id: UUID of the matter
        action: Short action description (e.g. "query", "contract_review", "draft")
        details: Optional longer description of what happened
        sources: Optional source reference string
        user: Who performed the action (default "System")
    """
    try:
        entry = AuditLog(
            id=uuid.uuid4(),
            matter_id=matter_id,
            action=action,
            user=user,
            details=details,
            sources=sources,
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log audit activity: {e}")
        try:
            db.rollback()
        except Exception:
            pass
