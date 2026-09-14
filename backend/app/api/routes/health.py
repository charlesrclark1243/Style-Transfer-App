from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    """
    Check the health of the application.

    Returns:
        Response: A simple status message.
    """

    return {"status": "ok"}
