# Slim Python image for minimal size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything to container
COPY . .

# Copy prefered model
COPY model_3epochs/ model_3epochs/


# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port that API uses
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
