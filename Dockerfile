FROM python:3

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY models/ ./models/
COPY snippets.json ./
COPY config.json ./
COPY langmain.py ./

EXPOSE 7860

CMD [ "python", "./langmain.py" ]