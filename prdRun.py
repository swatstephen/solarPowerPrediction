from 比對發電與氣象ver70 import goPrd
import schedule
import time

goPrd()
schedule.every().day.at("10:00").do(goPrd)

while True:
   schedule.run_pending()
   time.sleep(1) 