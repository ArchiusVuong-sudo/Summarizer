# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose any ports the app is expected to use (if any)
# EXPOSE 8080

# Define environment variables if needed
# ENV VAR_NAME=value

# Command to run your application
# You can adjust this based on whether you're running main.py or main.ipynb as a script
CMD ["python", "main.py"]
