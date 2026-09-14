from fastapi import APIRouter

from app.api.routes import stylyze

router = APIRouter()


@router.get("/health")
def health():
    """
    Check the health of the application.

    Returns:
        Response: The application status and whether the style transfer model is loaded.
    """

    return {"status": "ok", "model_loaded": stylyze.model is not None}
