import pandas as pd
from prophet import Prophet

## READING IN OUR CSV FILE TO OUR PANDAS DATAFRAME
read = pd.read_excel("/Users/nicholaskenney/PycharmProjects/Stop_Shop_FB_Profit_Module_Trial/ir211wk12samplefinal.xlsx",
                     sheet_name="Peyton_Manning_Views_Wiki")
df = pd.DataFrame(read)

## RENAMING COLUMN NAMES TO MATCH PROPHET FORMATTING
df.columns = ['ds', 'y']

## TRAIN MODEL: INITITIALIZING MODEL AND SENDING DF THROUGH PROPHET MODEL
m = Prophet()
m.fit(df)

## FORECAST AWAY
future = m.make_future_dataframe(periods=365)
# print(future.tail())

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(forecast.tail())
## PLOTTING FORECAST
fig1 = m.plot(forecast)

## PLOTTING AND DISPLAYING ALL COMPONENTS THAT GO INTO FORECAST
