# Boolean Reservoir
A simple boolean reservoir performing path integration.

## Installation

### Note
Currently, the installation process is focused on using DevContainers with Visual Studio Code. This approach utilizes a Conda image from Docker, which can be interactively run using VS Code's DevContainer stack, which should "just run". 

[Read more about devcontainers here](https://code.visualstudio.com/docs/remote/containers).

Optionally, there is a Conda environment file located in the `/docker/src` directory, but setting it up may require additional steps.


### Steps
0. **Prerequisite: Ensure you have Docker and Visual Studio Code installed:**
   - Install [Visual Studio Code](https://code.visualstudio.com/).
   - Install [Docker](https://docs.docker.com/get-started/get-docker/).

1. **Build the Docker image:**
   - Navigate to the `docker` folder.
   - Run the `setup.sh` script with the following command:
   ```bash
   . ./setup.sh
   ```
   **Note**: You may need to use `sudo` if you do not have passwordless Docker configuration.

2. **VS code:**
   - Now you should be able to open the repository in VS Code 
   - Install recommended devcontainer add-ons 
   - Atm there are no tutorials but you can now tinker with the project

2. **Github code:**
   git clone --recursive https://github.com/chrisvibe/boolean_reservoir.git

### Testing
Examples:
pytest /code/project/boolean_reservoir/test/test_graphs.py
pytest /code/project/path_integration/test/test_load_and_save.py
pytest /code/project/path_integration/test/test_verification_models.py
