# 1. Match your local Python version (3.12)
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for OpenCV 
# We use 'libgl1' because 'libgl1-mesa-glx' is outdated in Python 3.12 images
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install your "Golden List" of Python packages
# (Added gunicorn to the end of this list)
RUN pip install --no-cache-dir \
    "flask==3.1.3" \
    "keras==3.14.0" \
    "numpy==2.0.0" \
    "opencv-python-headless==4.10.0.84" \
    "pillow==11.0.0" \
    "tensorflow-cpu==2.18.0" \
    "gunicorn==23.0.0"

# 5. Set environment variable to force Keras 3 behavior
ENV KERAS_BACKEND="tensorflow"

# 6. Copy your project files into the container
COPY . .

# 7. Expose the Flask port
EXPOSE 5000

# 8. Start the application using Gunicorn instead of raw python
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
