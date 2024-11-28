import requests

url = 'http://localhost:5000/predict'
files = {'image': open('positive.jpeg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
files = {'image': open('negative.jpeg', 'rb')}
response = requests.post(url, files=files)
print(response.json())