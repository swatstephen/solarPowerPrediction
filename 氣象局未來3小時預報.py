import requests
import random
import numpy as np
from datetime import datetime

def info():
    weather = {'陰時多雲短暫陣雨或雷雨':'陰有雨','陰短暫陣雨或雷雨':'陰有雨','多雲時晴':'多雲','陰短暫雨':'有雨','陰時多雲短暫雨':'陰有雨','多雲':'多雲','多雲時陰':'多雲','晴時多雲':'晴','多雲午後短暫雷陣雨':'多雲有雷','陰時多雲':'陰天','陰天':'陰天', '陰時多雲陣雨或雷雨':'陰有雨','陰時多雲短暫陣雨':'陰有雨'}

    url = 'https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001?Authorization=CWA-23A85806-5FF0-4163-A006-FE38B48F7500'


    data = requests.get(url, timeout=10)

    data = data.json()

    #print(data['records']['location'][0]['locationName'])

    #hrs = [x for x in range(8,17)]
    hrs = [x for x in range(7,18)]

    #print(hrs)
    datas = []    
    j = 0
    k = 0
    for i in range(len(data['records']['location'])):
        #print(data['records']['location'][i]['locationName'])
        if data['records']['location'][i]['locationName'] == '高雄市':
            print(f"地區:{data['records']['location'][i]['locationName']}")
            print(f"起:{data['records']['location'][i]['weatherElement'][0]['time'][2]['startTime']}")
            print(f"終:{data['records']['location'][i]['weatherElement'][0]['time'][2]['endTime']}")
            print(f"氣象:{data['records']['location'][i]['weatherElement'][0]['time'][2]['parameter']['parameterName']}")
            print(f"最高溫:{data['records']['location'][i]['weatherElement'][4]['time'][2]['parameter']['parameterName']}")
            print(f"最低溫:{data['records']['location'][i]['weatherElement'][2]['time'][2]['parameter']['parameterName']}")
            wea = data['records']['location'][i]['weatherElement'][0]['time'][2]['parameter']['parameterName']
            dt = data['records']['location'][i]['weatherElement'][0]['time'][2]['startTime']
            maxs = data['records']['location'][i]['weatherElement'][4]['time'][2]['parameter']['parameterName']
            mins = data['records']['location'][i]['weatherElement'][2]['time'][2]['parameter']['parameterName']
            #以12點為最高溫，上午6個時段, 下午5個時段
            atom = np.linspace(float(mins),float(maxs),6)
            mtoa = np.linspace(float(maxs),float(mins),5)
            for i in range(len(hrs)):
                items = []
                #items.append(random.choice(['多雲','陰']))
                items.append(weather[wea])
                items.append(hrs[i])
                if hrs[i] <13:
                    items.append(atom[j])
                    j+=1
                else:
                    items.append(mtoa[k])
                    k+=1
                #items.append('bins')
                items.append(f'{hrs[i]}')
                datas.append(items)

    #print(datas)
    return datas, dt[:10], wea, maxs, mins
#a,b,c,d,e = info()
#print(c,d,e)