FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl \
    ffmpeg tesseract-ocr \
    libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libappindicator3-1 libasound2 libxtst6 libxrandr2 \
    chromium chromium-driver \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker will cache this layer unless requirements.txt changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the full app
COPY . .

# Set environment
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["python", "main.py"]
