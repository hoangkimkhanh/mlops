# Run app
## Run bash_script.sh 
bash bash_scipt.sh
## Run Makefile
make all

# Dockerfile-app
docker build -t fastapi-app -f Dockerfile-app .
docker run -p 30000:30000 fastapi-app

# Dockerfile-jenkins
docker build -t jenkins-image -f Dockerfile-jenkins .
## Push to Docker Hub
docker login
docker info
docker tag jenkins-image your-username/jenkins-image
docker push your-username/jenkins-image