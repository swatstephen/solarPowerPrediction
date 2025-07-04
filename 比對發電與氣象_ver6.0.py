import pandas as pd
import sqlite3
from sklearn.preprocessing import LabelEncoder
from 取出氣象資料函式 import show_list
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from 輸入介面 import enc
from 氣象局未來3小時預報 import info
from 連結LineNotyfy import linenotify


#讀入太陽能CSV資料及氣象的DB
df_solar = pd.read_csv('data.Csv')
df_weather = show_list()

#將兩張表的時間顯示相同
df_weather['時間'] = pd.to_datetime(df_weather['時間']).dt.strftime('%Y/%m/%d %H:%M:%S')
#print(df_weather)

#合併兩張表的資料
df_merged = pd.merge(df_solar, df_weather, on = '時間', how = 'inner')

#刪除NAN
df = df_merged.dropna()

#只選要顯示的欄位
df =  df[['時間','總發電量','氣象','外氣']]
df_origin = df[['時間','總發電量','氣象','外氣']]
#取出時間的小時為時段
df['時間'] = pd.to_datetime(df['時間'])
df['時段'] = df['時間'].dt.hour
print(df['時段'].describe()) #檢查時段的區間
#print(df)

#把時段從小時變成區間

#bins = [5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,15.5,16,19]
bins = [5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,19]
labels  = ['a','b','c','d','e','f','g','h','i','j']
df['時段'] = pd.cut(df['時段'], bins=bins, labels=labels)
print(df)
#df.dropna(subset=['時段']) #濾掉空值


#氣象資料hotenconding，分成5欄
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(df[['氣象']])
weather_encoded = onehot_encoder.transform(df[['氣象']]).toarray()


#編碼與原始資料對照
ecoColumn = onehot_encoder.get_feature_names_out(['氣象'])
showWea = pd.DataFrame(df['氣象'],columns=ecoColumn)
print(showWea)
#print(weather_encoded)

df[['wea1','wea2','wea3','wea4','wea5','wea6']] = weather_encoded

#時段資料hotencoding
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(df[['時段']])
time_encoded = onehot_encoder.transform(df[['時段']]).toarray()

#編碼與原始資料對照
ecoColumn = onehot_encoder.get_feature_names_out(['時段'])
showTim = pd.DataFrame(df['時段'],columns=ecoColumn)
print(showTim)
print(showWea)

ls = []
for i in range(1,11):
    ls.append(f'T{i}')
df[ls] = time_encoded
print(df)




#刪除多餘欄位
#df = df.drop(['氣象','wea5','時間','時段','T5'], axis = 1)
df = df.drop(['氣象','時間','時段','wea6','T10'], axis = 1)

#print(df)
#print(df.corr())


x = df[['外氣','wea1','wea2','wea3','wea4','wea5','T1','T2','T3','T4','T5','T6','T7','T8','T9']]
#x = df[['wea1','wea2','wea3','wea4','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
y = df['總發電量']

print(x,y)


#分成訓練集及測試集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=89)
#print(x_train, x_test, y_train, y_test)

#轉成numpy
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

#特徵縮放
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#print(len(y),len(y_train),len(y_test))
#print(len(y),len(y_train),len(y_test))
#print(x_train, x_test)

#y = w1*外氣+ w2*wea1....+b
w = np.array([x for x in range(1,16)])
b = 1
y_pred = (x_train * w).sum(axis = 1) + b

#cost function
#print(((y_train - y_pred)**2).mean())

def compute_cost(x, y, w, b):
    y_pred = (x * w).sum(axis = 1) + b
    cost = ((y - y_pred)**2).mean()
    return cost

ans  = compute_cost(x_train, y_train, w ,b)
#print(ans)

#optimizer

y_pred = (x_train * w).sum(axis = 1) + b
b_gradient = (y_pred - y_train).mean()
w_gradient = np.zeros(x_train.shape[1])

for i in range(x_train.shape[1]):
    w_gradient[i] = ((x_train[:,i]) * (y_pred-y_train)).mean()

#print(w_gradient, b_gradient)

def compute_gradient(x, y, w, b):
    y_pred = (x * w).sum(axis = 1) + b
    w_gradient = np.zeros(x.shape[1])
    b_gradient = (y_pred - y).mean()
    for i in range(x.shape[1]):
        w_gradient[i] = ((x[:,i]) * (y_pred-y)).mean()

    return w_gradient, b_gradient

a1, b1 = compute_gradient(x_train, y_train, w, b)
#print(a1,b1)

learning_rate = 0.001
w_gradient, b_gradient = compute_gradient(x_train, y_train, w, b)

#print(compute_cost(x_train, y_train, w, b))
w = w - w_gradient*learning_rate
b = b - b_gradient*learning_rate
#print(compute_cost(x_train, y_train, w, b))

#計算

np.set_printoptions(formatter={'float': '{:2e}'.format})
def gradient_decent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter =1000):

  c_hist = []
  w_hist = []
  b_hist = []

  w = w_init
  b = b_init

  for i in range(run_iter):
      w_gradient, b_gradient = gradient_function(x, y, w, b)

      w = w-w_gradient * learning_rate
      b = b-b_gradient * learning_rate
      cost = cost_function(x, y, w, b)

      w_hist.append(w)
      b_hist.append(b)
      c_hist.append(cost)

      if i%p_iter == 0:
        print(f'Iteration {i:5} : cost {cost: .4e},  w:{w},  b:{b: .4e},  w_gradient:{w_gradient},  b_gradient:{b_gradient: .4e}')

  return w, b, w_hist, b_hist, c_hist

w_init = np.array([x for x in range(1,16)])
b_init = 0
learning_rate = 0.0022
run_iter = 5000

w_final, b_final, w_hist, b_hist, c_hist = gradient_decent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter)

print(w_final,b_final)

y_pred = (w_final * x_test).sum(axis = 1) + b_final
df = pd.DataFrame({
    'y_pred': y_pred,
    'y_test': y_test,
})

print(df)
df = df.join(df_origin, how = 'inner')
df.to_csv('預測與實際值比對.csv',encoding='utf-8-sig')

print(compute_cost(x_test, y_test, w_final, b_final))

#預測數據

#datas = [
#    ['陰',8,25.5,'08:00'],
#    ['多雲',9,27,'09:00'],
#    ['多雲',10,28,'10:00'],
#   ['多雲',11,29.5,'11:00'],
#    ['多雲',12,31,'12:00'],
#    ['多雲',13,31,'13:00'],
#    ['多雲',14,31,'14:00'],
#    ['多雲',15,31,'15:00'],
#    ['多雲',16,29.5,'16:00'],
#    ]

datas,dt = info()

def predict(a,b,c):
    data = enc(a,b,c)
    #data = enc('多雲',16, 25, bins)

    #[5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,17]Columns: [a,b,c,d,e,f,g,h,i,j,k]
    #[氣象_多雲, 氣象_多雲有雷, 氣象_晴, 氣象_陰, 氣象_陰有雨]  

    #實際應用  #10/19 外氣25.8 陰 [0,0,1,0] 下午一點[0,0,0,1,0,0,0,0,0,0]
    x_real = np.array([data])
    x_real = scaler.transform(x_real)
    y_real = (w_final * x_real) . sum(axis = 1) +b_final
    #print(y_real)
    return y_real
print(f'預測日{dt}')
tt = 0
dday = []
dtime = []
dpower = []

for d in datas:
    ns = predict(d[0],d[1],d[2])
    ns = np.around(ns, decimals=1)
    print(f'{d[3]}時預測時發電{ns[0]}度')
    dday.append(dt)
    dtime.append(str(d[3].zfill(2)))
    dpower.append(ns[0])
    tt += ns
#tt = np.around(tt, decimals=1)    
print(f'今天總發電預測{tt[0]:.2f}度')
print(dday,dtime,dpower)
df = pd.DataFrame({'日期':dday, '時段':dtime, '預測發電(度)':dpower})
print(df)
df.to_csv('預測記錄.csv',encoding='utf-8-sig',mode='a', index=False)
df = df.to_string(index=False)
linenotify(df)
