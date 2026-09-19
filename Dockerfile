FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MEETING_FFMPEG_BIN=/usr/bin/ffmpeg
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && test -x /usr/bin/ffmpeg \
    && /usr/bin/ffmpeg -version

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && python -c "import firebase_admin; print(f'firebase-admin={firebase_admin.__version__}')" \
    && test -x /usr/bin/ffmpeg

COPY . .

EXPOSE 8080

CMD ["uvicorn", "apps.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
