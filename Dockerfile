FROM python:3.11

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the application code
ENV PYTHONPATH=/app

COPY . .

# Expose the port
EXPOSE 5000

# Run the Flask app
CMD ["python", "run.py"]