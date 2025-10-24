# Đồng bộ nền tảng với DeepStream để tránh lệch libc/cuda
ARG DEEPSTREAM_TAG=6.3-triton-multiarch
FROM nvcr.io/nvidia/deepstream:${DEEPSTREAM_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Tránh các wheel khó trên ARM64 nếu vô tình có trong requirements
RUN sed -i '/opencv-python/d' requirements.txt || true \
 && sed -i '/onnxruntime/d' requirements.txt || true \
 && sed -i '/torch/d' requirements.txt || true \
 && sed -i '/tensorflow/d' requirements.txt || true

RUN python3 -m pip install --upgrade pip==24.2 wheel \
 && python3 -m pip install -r requirements.txt \
 || true \
 && python3 -m pip install "numpy<2" uvicorn[standard] fastapi requests || true

COPY . /app
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
