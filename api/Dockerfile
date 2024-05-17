# Dockerfile

FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container at /app
COPY ../ /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Run api.py when the container launches
CMD ["python", "api/api.py"]
