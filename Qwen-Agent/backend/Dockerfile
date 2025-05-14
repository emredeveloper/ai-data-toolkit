FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7864

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7864"]