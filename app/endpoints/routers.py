from fastapi import APIRouter
from app.endpoints import ocr

api_router = APIRouter()
api_router.include_router(ocr.router)