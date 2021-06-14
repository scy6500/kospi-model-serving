FROM python:latest

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "serving.py"]