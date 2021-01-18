import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

scaler=MinMaxScaler(feature_range=(0,1))

df=pd.read_csv("NSE-TATA.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Closing Prices (Validation)')

data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values

train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

lstm_model.save("saved_lstm_model.h5")

train_data=new_dataset[:987]
valid_data=new_dataset[987:]

print("valid data")
print(valid_data)
print("\nnum predictions", len(predicted_closing_price))

valid_data.insert(1, "Predictions", predicted_closing_price)

print("updated valid data")
print(valid_data)

print("training dataset")
print(train_data)

mse = mean_squared_error(valid_data['Close'], valid_data['Predictions'], squared=False)
rmse = sqrt(mse)

print("mse: ", mse)
print("rmse: ", rmse)

# valid_data.loc[:,('Predictions')]=predicted_closing_price
plt.plot(train_data["Close"], label="Closing Prices (Training)")
plt.title("Closing Price Predictions")
plt.xlabel("Year")
plt.ylabel("Stock Price")
plt.legend()
plt.show(block=True) #enable plot gen when executing from shell
plt.plot(valid_data[['Close']], label="Actual Closing Prices")
plt.plot(valid_data['Predictions'], label="Predicted Closing Prices")
plt.title("Closing Price (Actual vs. Predicted)")
plt.xlabel("Year")
plt.ylabel("Stock Price")
plt.legend()
plt.show(block=True) #enable plot gen when executing from shell

# show actual vs predicted in an overall plot
plt.plot(df["Close"],label='Actual Closing Prices')
plt.plot(valid_data['Predictions'], label="Predicted Closing Prices")
plt.title("Closing Price (Actual vs. Predicted)")
plt.xlabel("Year")
plt.ylabel("Stock Price")
plt.legend()
plt.show(block=True) #enable plot gen when executing from shell
