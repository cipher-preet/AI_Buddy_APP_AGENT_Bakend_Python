from fastapi import APIRouter, UploadFile, File
from apps.api_gateway.controllers.speech_controller import (
    transcribe_audio_controller
)
router = APIRouter()

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    return await transcribe_audio_controller(file)