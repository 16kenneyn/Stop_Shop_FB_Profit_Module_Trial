import pandas as pd
from fbprophet import Prophet
import datetime as date
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
## READING IN OUR CSV FILE TO OUR PANDAS DATAFRAME
read = pd.read_excel("/Users/nicholaskenney/PycharmProjects/Stop_Shop_FB_Profit_Module_Trial/Pre_Prediction_Data.xlsx", sheet_name="Hormel_Salami")
df = pd.DataFrame(read)

'''
## CONVERTING OUR DATE FROM DATABLOCK TO DATETIME TYPE.  USING LAMBDA TO ITERATE THROUGH EACH VALUE IN DATE.
df["Year"] = df["OrderDate"].apply(lambda x: "20" + str(x)[-2:])
df["Month"] = df["OrderDate"].apply(lambda x: str(x)[:2])
df["Day"] = df["OrderDate"].apply(lambda x: str(x)[-5:-3])
df['ds'] = pd.DatetimeIndex(df['Year'] + '-' + df['Month'] + '-' + df['Day'])  #FORMATS NEW DATE TYPE TO MATCH PROPHET USE.
'''

## DROPPING ALL COLUMNS OTHER THAN DATE AND SALES
#df.drop(["OrderDate", "Region", 'Year', 'Month', 'Day'], axis=1, inplace=True)
#df.drop(["Region"], axis=1, inplace=True)

## RENAMING COLUMN NAMES TO MATCH PROPHET FORMATTING
df.columns = ['ds', 'y']

## TRAIN MODEL: INITITIALIZING MODEL AND SENDING DF THROUGH PROPHET MODEL
m = Prophet(weekly_seasonality=True)
model = m.fit(df)

## FORECAST AWAY
future = m.make_future_dataframe(periods=52, freq='W')
forecast = m.predict(future)


## PLOTTING FORECAST
#forecast[['ds', 'yhat']]
plot1 = plot_plotly(m, forecast)
plt.show()
## PLOTTING AND DISPLAYING ALL COMPONENTS THAT GO INTO FORECAST

plt2 = m.plot_components(forecast)
plt.show()

#export = input("Would you like to export to Excel?")

## EXPORTING FORECAST DATA TO EXCEL FILE
#if export == "yes" or "Yes" or "YES" or "Yeah" or "yea" or "Ye":
forecast.to_excel("/Users/nicholaskenney/PycharmProjects/Stop_Shop_FB_Profit_Module_Trial/Exported_Excel_Files/Prediction_Results.xlsx", sheet_name="Hormel_Salami")

