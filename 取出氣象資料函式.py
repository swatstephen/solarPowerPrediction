import pandas as pd
import sqlite3

def show_list(): 
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
    #print(dt,wa,wd,ws,ait,rh,ap)
    #df = pd.DataFrame({'時間':dt,'氣象':wa,'風向':wd,'風速':ws,'外氣':ait,'濕度':rh,'氣壓':ap})
    df = pd.DataFrame({'時間':dt,'氣象':wa,'風向':wd,'風速':ws,'外氣':ait,'濕度':rh,'氣壓':ap})
    return df

#print(show_list())