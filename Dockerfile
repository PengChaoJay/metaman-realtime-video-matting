ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-model.txt ./
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url ${PIP_INDEX_URL} -r requirements.txt \
    && pip install --no-cache-dir --timeout 300 --retries 10 --no-deps \
    --index-url ${PIP_INDEX_URL} -r requirements-model.txt

COPY requirements-model-deps.txt ./
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url ${PIP_INDEX_URL} -r requirements-model-deps.txt

COPY app ./app
COPY static ./static

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]

