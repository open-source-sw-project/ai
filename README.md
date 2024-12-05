#docker 사용법

1. docker build -t 
2. 로컬에서 테스트
     - test폴더에 이미지를 저장
     - test_api.py 파일의 파일 경로 수정
     - python test_api.py 실행
4. docker-compose up --build 후 동일한 테스트 실행


#kubernetes 사용법

1. minikube start
2. kubectl apply -f flask-deployment.yaml
3. minikube dashboard로 상태 확인 안되면 >> minikube addons enable dashboard 후 minikube dahsboard
4. 포트포워딩: kubectl port-forward service/handoc-flask-service 8080:80
5. http:\\localhost:8080\health로 서버 동작상태 확인 가능
