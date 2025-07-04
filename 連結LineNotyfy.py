import requests
from datetime import datetime
current_datetime = datetime.now()
'''
url = 'https://notify-api.line.me/api/notify'
token = '6bn3eM2i5WkfSij9fS3bHlxPWjkSDVhshUFGcxGmj6q'
headers={'Authorization':f'Bearer {token}'}
payload = {
            'message': 'hello',
        }
'''
def linenotify(prd,wea,mx,mi):
    url = 'https://notify-api.line.me/api/notify'
    token = 'IRWo44paU4LQRC17GOkonoN9SMOK8AekiFJVNyOI85s'
    headers={'Authorization':f'Bearer {token}'}
    payload = {
            'message': f'送出時間：{current_datetime} \n {prd} \n 預測天氣：{wea} \n 最高氣溫:{mx} \n 最低氣溫:{mi}',
        }
    response = requests.post(url,headers=headers,data=payload)
    return response.status_code,response.text

def linenotify2(msg):
    url = 'https://notify-api.line.me/api/notify'
    token = '6bn3eM2i5WkfSij9fS3bHlxPWjkSDVhshUFGcxGmj6q'
    headers={'Authorization':f'Bearer {token}'}
    payload = {
            'message': msg,
        }
    response = requests.post(url,headers=headers,data=payload)
#x , y = linenotify(url,token,headers,payload)
#print(linenotify(url,token,headers,payload))
#linenotify()