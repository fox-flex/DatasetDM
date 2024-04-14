# Use the official CUDA 11.1 base image
# FROM cnstark/pytorch:1.9.1-py3.9.12-cuda11.1.1 -ubuntu20.04
FROM cnstark/pytorch:1.10.1-py3.9.12-cuda11.1.1-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# RUN pip --version
# RUN pip3 --version
# RUN python -c "import torch; print(torch.__version__)"
# RUN python3 -c "import torch; print(torch.__version__)"

# RUN exit 1
RUN apt update && \
    apt install -y \
        git wget \
        ffmpeg libsm6 libxext6

# Set the working directory
WORKDIR /app
RUN pip3 install --upgrade pip
RUN wget https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/detectron2-0.6%2Bcu111-cp39-cp39-linux_x86_64.whl && \
    pip3 install ./detectron2-0.6+cu111-cp39-cp39-linux_x86_64.whl
RUN pip3 install diffusers==0.3.0
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# # Install gdown and download the file
# RUN pip3 install gdown
# RUN gdown 12lrOexljsyvFB30-ltbYXnIpQ8oP4lrW

# Change back to the root directory
RUN rm -rf requirements.txt detectron2-0.6+cu111-cp39-cp39-linux_x86_64.whl
# WORKDIR /app

# Install huggingface/diffusers version 0.3.0
# RUN pip3 install transformers==4.11.3

# Set the entrypoint command
# CMD ["python3", "your_script.py"]