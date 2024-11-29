#docker 사용법

1. 로컬에서 테스트 -> 아무 사진이나 가져다 놓고 'localhost:5000/predict'로 전송
2. 잘 되면 docker-compose up --build 후 동일한 테스트 실행
3. 그것도 잘 되면 ok


#kubernetes 사용법

1. minikube start
2. kubectl apply -f flask-deployment.yaml
3. minikube dashboard 안되면 >> minikube addons enable dashboard 후 minikube dahsboard
4. kubectl port-forward service/handoc-flask-service 5000:80
5. http:\\localhost:5000\health로 서버 동작상태 확인 가능
