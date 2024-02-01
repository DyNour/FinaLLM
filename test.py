import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
validation_loss=[]
stock_symbol = "AAPL"
start_date = "1980-01-01"
end_date = "2023-12-31"
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
"""print(stock_data.head())

print(len(stock_data))"""

columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
data = stock_data[columns_to_use]

scaler = MinMaxScaler(feature_range=(0.1, 1))
data_scaled = scaler.fit_transform(data)

scaler_y = MinMaxScaler(feature_range=(0.1, 1))
y = scaler_y.fit_transform(data[['Adj Close']])

X = [data_scaled[i][:5] for i in range(len(data_scaled))]

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(Dense(units=6, activation='tanh', input_dim=5))
"""model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=2, activation='relu'))"""
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=16,validation_data=(X_test, y_test))

predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

train = stock_data[:len(stock_data)-len(X_test)]
test = stock_data[len(stock_data)-len(X_test):]
test['Predictions'] = predictions

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss ')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
"""plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('ADJ Close (USD)')
plt.plot(train['Adj Close'])
plt.plot(test[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Actual Test', 'Predictions'])"""
plt.show()
