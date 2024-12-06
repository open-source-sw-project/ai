#docker 사용법
1. docker pull 2jay0/handoc-flask:latest
2. docker build -t 
3. 로컬에서 테스트
     - test폴더에 이미지를 저장
     - test_api.py 파일의 파일 경로 수정
     - python test_api.py 실행
4. docker-compose up --build 후 동일한 테스트 실행


#kubernetes 사용법(로컬 테스트용입니다.)
1. minikube start
2. kubectl apply -f flask-deployment.yaml (빌드타임 약 40분 소요)
3. minikube dashboard로 상태 확인 안되면 >> minikube addons enable dashboard 후 minikube dahsboard
4. minikube service handoc-flask-service --url 명령으로 현재 제공 IP확인
5. http:\\localhost:(확인된 포트번호)\health로 서버 동작상태 확인 가능
