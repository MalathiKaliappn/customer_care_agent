# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY ./app ./app

# CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.enableCORS=false"]

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY . .

# Default command for Streamlit
CMD ["streamlit", "run", "src/pages/main.py", "--server.port=8501", "--server.enableCORS=false"]
