FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY ml ./ml
COPY models ./models
COPY app.py ./app.py
COPY train.py ./train.py
COPY predict.py ./predict.py
COPY evaluate.py ./evaluate.py
RUN mkdir -p temp_uploads
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
