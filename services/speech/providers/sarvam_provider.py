import httpx
from apps.api_gateway.config.setting import settings

print("this is api ket present", settings.SARVAM_API_KEY)

async def sarvam_transcribe(file):

    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": settings.SARVAM_API_KEY
    }

    files = {
        "file": (
            file.filename,
            await file.read(),
            file.content_type
        )
    }

    async with httpx.AsyncClient() as client:

        response = await client.post(
            url,
            headers=headers,
            files=files
        )

    return response.json()