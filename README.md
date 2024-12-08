# Dockerfile-app
docker build -t fastapi-app -f Dockerfile-app .
docker run -p 30000:30000 fastapi-app

# Dockerfile-jenkins
docker build -t jenkins-image -f Dockerfile-jenkins .
## push to Docker Hub
docker login
docker info
docker tag jenkins-image your-username/jenkins-image
docker push your-username/jenkins-image