FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

COPY . .

RUN echo "Downloading model for RUN_ID=${RUN_ID}"

CMD ["echo", "Container ready"]