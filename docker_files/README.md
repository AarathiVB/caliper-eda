# Caliper Streamlit App Docker Image Creation

This guide covers the steps to create a Docker image for the Caliper Streamlit application, run it locally, and save the image as a TAR file.

## Prerequisites

- Docker installed on your system
- Dockerfile and requirements.txt for the Caliper Streamlit app

## Steps

1. **Build the Docker Image**:
   Open a terminal (or Command Prompt on Windows, Terminal on Mac) and navigate to the directory containing both your Dockerfile.txt and requirements.txt, then execute the following command:
   
   ```
   docker build -t caliper_streamlit_app -f Dockerfile.txt .
   ```


2. **Run the Docker Container**:
   To start the application on your local machine, run:

   ```
   docker run -p 8501:8501 caliper_streamlit_app
   ```

   The app will be accessible at `http://localhost:8501`.

3. **Save the Docker Image**:
   Save the Docker image as a TAR file with the command:

   ```
   docker save caliper_streamlit_app > caliper_streamlit_app.tar
   ```

## Usage

After completing these steps, you will have a Docker container running the Caliper Streamlit application accessible through your web browser, and the Docker image saved as a TAR file for distribution or backup purposes.


