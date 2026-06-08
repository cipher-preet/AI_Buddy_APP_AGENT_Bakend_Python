import httpx
from apps.api_gateway.config.setting import settings


async def sarvam_transcribe_from_path(
    file_path: str,
    filename: str,
    content_type: str,
):
    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": settings.SARVAM_API_KEY,
    }

    data = {
        "model": "saarika:v2.5",
        "language_code": "unknown",
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (
                filename,
                audio_file,
                content_type or "audio/wav",
            )
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers=headers,
                data=data,
                files=files,
            )

    if response.status_code >= 400:
        print("Sarvam status:", response.status_code)
        print("Sarvam error:", response.text)

    response.raise_for_status()
    return response.json()