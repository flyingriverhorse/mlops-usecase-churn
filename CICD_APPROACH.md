# CI/CD Approach

### 1. Continuous Integration (CI)
*Triggered on every Pull Request.
- **Code Formatting**: Run `black --check .` to ensure code style compliance (88 chars).
- **Quality Check**: Run linters (`flake8`) to locate logical errors.
- **Automated Tests**: Execute `pytest` to validate API logic and data schemas.
- **Security Audit**: Run `safety check` to scan dependencies for known vulnerabilities.
- **Sanity Check**: Verify that the latest trained model (`churn_model.pkl`) loads correctly.

### 2. Continuous Deployment (CD)
*Triggered on merge to Main.

The deployment process follows a "Build-Push-Deploy" strategy using a container registry (Docker Hub).

- **Build & Push**:
    1. **Log in**: GitHub Actions authenticates with Docker Hub using secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`).
    2. **Build**: Docker image is built from the `Dockerfile` with the tag `latest`.
    3. **Push**: The built image is pushed to the Docker Hub repository (most likely for production use can be other platforms).

- **Deploy (SSH)**:
    1. **Connect**: CI/CD pipeline connects to the production server via SSH.
    2. **Update**: Executes `docker-compose pull` to fetch the new image from Docker Hub.
    3. **Restart**: Runs `docker-compose up -d --force-recreate` to switch to the new container.

- **Health Verification**:
    - The pipeline waits 30 seconds for containers to stabilize.
    - Executes `curl -f http://localhost:8000/health` to verify the API is responsive.
