
import re 
import streamlit as st 
import pandas as pd 
import altair as alt

df = pd.read_csv('supermarket_sales.csv')

date = []
week = []

for c in df['Date']:
    c = re.findall(r'\d+', c)
    m, d = int(c[0]), int(c[1])

    if m == 3:
        d += (31+28)
    
    elif m == 2:
        d += 31

    d += -1


    # 2019年1月1日は火曜日
    if d%7 == 6: tmp = 'Mon'
    elif d%7 == 0: tmp = 'Tue'
    elif d%7 == 1: tmp = 'Wed'
    elif d%7 == 2: tmp = 'Thr'
    elif d%7 == 3: tmp = 'Fri'
    elif d%7 == 4: tmp = 'Sat'
    elif d%7 == 5: tmp = 'Sun'
    date.append(tmp)

    if d//14 == 0: tmp = '1/1~1/14'
    elif d//14 == 1: tmp = '1/15~1/28'
    elif d//14 == 2: tmp = '1/29~2/11'
    elif d//14 == 3: tmp = '2/12~2/25'
    elif d//14 == 4: tmp = '2/26~3/11'
    elif d//14 == 5: tmp = '3/12~3/25'
    elif d//14 == 6: tmp = '3/26~3/31' # 最終期は他より短いことに注意
    week.append(tmp)

df['day'] = date
df['week'] = week

time = []
for c in df['Time']:
    c = re.findall(r'\d+', c)
    t = c[0] + ':00~' + c[0] +':59'
    time.append(t)

df['time'] = time

st.table(df.head(10))
