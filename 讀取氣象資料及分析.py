import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.font_manager import fontManager
import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

fontManager.addfont('ChineseFont.ttf')
mlp.rc('font', family='ChineseFont')

conn = sqlite3.connect('D:\太陽能AI\大社_橋頭_阿公店測站.db')
cursor = conn.cursor()
cursor.execute('select * from 小時資料')
rows = cursor.fetchall()
dt = []
wa = []
wd = []
ws = []
ait = []
rh = []
ap = []

for row in rows:
    if row[1] == '大社':
        if row[3] != -99 and row[8] >900 :
            dt.append(row[2])
            wa.append(row[3])
            wd.append(row[4])
            ws.append(row[5])
            ait.append(row[6])
            rh.append(row[7])
            ap.append(row[8])
print(dt,wa,wd,ws,ait,rh,ap)
#df = pd.DataFrame({'時間':dt,'氣象':wa,'風向':wd,'風速':ws,'外氣':ait,'濕度':rh,'氣壓':ap})
df = pd.DataFrame({'時間':dt,'氣象':wa,'風向':wd,'風速':ws,'外氣':ait,'濕度':rh,'氣壓':ap})
print(df)
le = LabelEncoder()
df['氣象'] = le.fit_transform(df['氣象'])
min_max = MinMaxScaler()
df[['風向','風速','外氣','濕度','氣壓']] = min_max.fit_transform(df[['風向','風速','外氣','濕度','氣壓']])
print(df)
df.set_index('時間',inplace=True)
#df.plot(df[['風速','外氣','濕度','氣壓']])
df[['風速','外氣','濕度','氣壓']].plot(kind = 'line',title = '大社氣象資訊')
df.plot.scatter(x = '風速', y= '氣壓' , title = '風速-氣壓關係圖')
df.plot.scatter(x = '外氣', y= '濕度' , title = '外氣-濕度關係圖')
df.plot.scatter(x = '風速', y= '濕度' , title = '風速-濕度關係圖')
df.plot.scatter(x = '風速', y= '外氣' , title = '風速-外氣關係圖')
df.plot.scatter(x = '氣壓', y= '外氣' , title = '氣壓-外氣關係圖')
#plt.title('大社氣象資訊')
plt.xlabel('時間')

plt.ylabel('Scale')
plt.xticks(rotation=90)
plt.show()


'''
fontManager.addfont('ChineseFont.ttf')
mlp.rc('font', family='ChineseFont')

df = pd.read_csv('小時資料.csv')
print(df)
df = df.rename(columns={'serial':'序','local':'地點','dt':'時間','weather':'氣象','WindDirection':'風向','WindSpeed':'風速','AirTemperature':'氣溫','RelativeHumidity':'相對濕度','AirPressure':'氣壓'})
print(df)

print(df[(df['地點']=='大社') & (df['氣象'] != '-99')])
print(df['時間'].str.slice(11,16))
'''
