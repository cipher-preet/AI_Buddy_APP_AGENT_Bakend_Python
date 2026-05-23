from services.speech.providers.sarvam_provider import (
    sarvam_transcribe
)
async def transcribe_audio_service(file):

    transcript = await sarvam_transcribe(file)

    return transcript