import requests
url = 'http://127.0.0.1:{your_node_port}/predict'
files = {'image': open('positive.jpeg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
files = {'image': open('negative.jpeg', 'rb')}
response = requests.post(url, files=files)
print(response.json())