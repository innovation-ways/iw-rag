# iw-rag Chainlit frontend — CPU-only, pure API client.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8501
# CHAINLIT_AUTH_SECRET must be set in the environment for password auth.
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8501", "--headless"]
