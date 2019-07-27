# 비트코인 미래시세 예측

from fbprophet import Prophet
import pandas as pd
dfs = pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20180101&end=20181231')
df = dfs[0]
# print(df.head())

# 시간 순 정렬

df = df.iloc[::-1]  # 시간 순으로 뒤집기
print(df.head())

df['ds'] = df['Date']
df['y'] = df['Close**']
forecast_data = df[['ds', 'y']].copy()
print(forecast_data.head())

m = Prophet()
m.fit(forecast_data)

future = m.make_future_dataframe(periods=90)
# print(future.tail())
forecast = m.predict(future)
# print(forecast)
print('2019-03-31 최저 최고 예측값 ==')
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())