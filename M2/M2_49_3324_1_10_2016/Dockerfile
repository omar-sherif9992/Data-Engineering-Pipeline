FROM python:3.11

WORKDIR /app

# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt


# Expose the desired port (if needed)
EXPOSE 8888

# Set the entrypoint
ENTRYPOINT ["python", "src/main.py"]
