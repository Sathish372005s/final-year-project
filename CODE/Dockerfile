# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask pandas scikit-learn

# Expose port (Flask default)
EXPOSE 5000

# Run app
CMD ["python", "app.py"]