# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# # Set working directory
# WORKDIR /app

# # Copy dependencies file and install packages
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt


# # Copy the entire source code into the container
# COPY . .

# # Default command for Streamlit
# CMD ["streamlit", "run", "src/ui/main.py", "--server.port=8501", "--server.enableCORS=false"]



FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy dependencies file and install packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY . .

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Start both Streamlit and FastAPI using a process manager (e.g., `tini` or `sh -c`)
# CMD sh -c "streamlit run src/ui/main.py --server.port=8501 --server.enableCORS=false & \
#           uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload"
CMD ["bash", "-c", "streamlit run src/ui/main.py --server.port=8501 --server.enableCORS=false & uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000"]

# CMD ["streamlit", "run", "src/ui/main.py", "--server.port=8501", "--server.enableCORS=false"]
