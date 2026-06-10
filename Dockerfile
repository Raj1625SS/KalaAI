FROM python:3.11-slim

WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY app.py .
COPY models/ models/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]