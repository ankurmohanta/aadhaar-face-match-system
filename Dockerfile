FROM ultralytics/ultralytics:latest
# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt
# Copy entire codebase into container
COPY . /workspace/
# Set working directory
WORKDIR /workspace
CMD ["bash"]