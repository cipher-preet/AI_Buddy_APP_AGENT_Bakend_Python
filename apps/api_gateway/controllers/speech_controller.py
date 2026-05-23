
from services.speech.speech_service import (
    transcribe_audio_service
)
async def transcribe_audio_controller(file):

    result = await transcribe_audio_service(file)
    
    print("------------->> ", result)

    return {
        "success": True,
        "data": result
    }