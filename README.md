# CALIPER

The CALIPER project is about creating a toolbox for exploratory analysis of experiment data.

## Installation

- [Python](https://www.python.org/downloads/) is the used programming language.
- [Anaconda](https://www.anaconda.com/download/) is the used python package management tool.
- [Anaconda Navigator](https://anaconda.org/anaconda/anaconda-navigator) is a graphical tool that helps with creating Anaconda Environments. Such environments are collections of packages in a specific version. 
- [Streamlit](https://docs.streamlit.io/library/get-started/installation) is the python framework with which the CALIPER web app is build.

## Development

### Concepts

Familiarize yourself with:
- [Streamlit Main Concepts](https://docs.streamlit.io/library/get-started/main-concepts)
- Basic structure:
    - root directory:
        - home_page.py (which is the home page of the web app)
        - pages directory:
            - data page (where to upload the experiment data)
            - analysis page (where to use the statistical functionalities)

### Git flow

When working on a new feature:
- create a branch of type "feature" from main
- implement feature
- test feature
- create merge request (PR) to main and ask for reviewer

## Usage

To start the streamlit app, you have to run the following command in the root directory of the project:

```
streamlit run Home.py
```


# Steps to use Caliper Streamlit Application

This manual provides instructions on how to install Docker and run the Caliper Streamlit application using the provided Docker image file `caliper_streamlit_app.tar`.

## Part 1: Installing Docker

### For Windows / Mac:

1. **Download Docker Desktop:**
   - Visit the [Docker Hub](https://www.docker.com/products/docker-desktop) and download the appropriate installer for your operating system.

2. **Install Docker Desktop:**
   - Run the installer and follow the on-screen instructions to install Docker Desktop on your system.

3. **Start Docker Desktop:**
   - Launch Docker Desktop from your Applications folder (Mac) or Start Menu (Windows). On the first launch, Docker Desktop may ask for additional permissions.

4. **Configure Docker Desktop (if necessary):**
   - Docker Desktop should work with default settings, but you can adjust resources (like CPU and memory) allocated to Docker via the settings/preferences menu if needed.

### For Linux:

1. **Update Package List:**
   - Open a terminal window and run the following command:
     ```
     sudo apt-get update
     ```

2. **Install Docker:**
   - Install Docker using the following command:
     ```
     sudo apt-get install docker-ce docker-ce-cli containerd.io
     ```

3. **Start Docker:**
   - You can start Docker with the following command:
     ```
     sudo systemctl start docker
     ```

4. **Enable Docker on Boot (optional):**
   - To make sure Docker starts when your system boots, run the following command:
     ```
     sudo systemctl enable docker
     ```

5. **Manage Docker as a non-root User (optional):**
   - To run Docker commands without `sudo`, add your user to the `docker` group using the following command:
     ```
     sudo usermod -aG docker your-user
     ```
   - Replace `your-user` with your username. Log out and log back in for this to take effect.

## Part 2: Loading and Running the Docker Image

1. **Receive the Docker Image File:**
   - Obtain the `caliper_streamlit_app.tar` file from the sender or download it from OVGU GitLab repo and save it to a known directory on your system.

2. **Load the Docker Image:**
   - Open a terminal (or Command Prompt on Windows, Terminal on Mac) and navigate to the directory where you saved the `caliper_streamlit_app.tar` file. 
     Run the following command to load the image into Docker:
     ```
     docker load < caliper_streamlit_app.tar
     ```

3. **Run the Docker Container:**
   - Start the container with the following command:
     ```
     docker run -p 8501:8501 caliper_streamlit_app
     ```
   - This command tells Docker to map port 8501 on your local machine to port 8501 inside the Docker container, making the app accessible via `http://localhost:8501`.

4. **Access the Streamlit Application:**
   - After running the container, open your web browser and navigate to `http://localhost:8501`. You should see the Caliper Streamlit app running.
   - Note:
A default Network URL and External URL will be given by Streamlit. If these URLs do not work, you can use `http://localhost:8501` to open Caliper Streamlit app.


### Additional Tips:

- Ensure Docker is running before attempting to load or run the Docker image.
- If there is an error related to port allocation (e.g., the port is already in use), you can map the Streamlit app to a different port by changing the first port number in the run command (e.g., `-p 8502:8501` to use port 8502 on the host).
- If you encounter any issues, refer to the official Docker documentation, the support forums, or contact the developers for troubleshooting assistance.
