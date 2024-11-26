#사용법

1. empty.json을 firebase -> 프로젝트 설정 -> 서비스 계정 -> 새 비공개키 생성 으로 만든 json파일로 교체
2. app.py의 18번줄 "firebase-account.json"파일을 교체한 파일로 변경, 28-30 줄은 주석처리해도 됨
3. 로컬에서 테스트 -> 아무 사진이나 가져다 놓고 'localhost:5000/predict'로 전송
4. 잘 되면 docker-compose up --build 후 동일한 테스트 실행
5. 그것도 잘 되면 ok
