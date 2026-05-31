# pytorch cuda base image that the build will extend
#FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ultralytics/ultralytics:latest-cpu

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# create and move terminal into '/app' directory. All future commands execute from '/app'
WORKDIR /app

# copy requirements.txt file to '/app' for docker caching
COPY requirements.txt .

# Execute and install all the required libraries in the container
RUN pip install --no-cache-dir -r requirements.txt

# copy entire project from local to '/app' in the container
COPY . .

# expose port
EXPOSE 8000

# start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
