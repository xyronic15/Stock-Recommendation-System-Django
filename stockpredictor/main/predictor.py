from gc import callbacks
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import callbacks
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
from .helper import get_ticker, get_past, CHART_HEIGHT, CHART_WIDTH
import ta
from math import isnan

PERIOD = "10y"
BUY, SELL, STAY = -1, 1, 0

# depth for determining previous body average
PREV_DEPTH_BODY_AVG = 14

# depth for determining the trend direction
PREV_DEPTH_TREND = 50

# depth for determining upcoming trend direction
FUTURE_DEPTH = 10

class predictor:

    def __init__(self, ticker):
        
        self.ticker = ticker
        self.stock = get_ticker(self.ticker)
        self.past = get_past(self.stock, PERIOD)
        self.close_data = self.past.filter(['Close'])
        self.predictions = None
        self.model = None
        self.test_fig = None
        self.predict_fig = None
    
    def train(self, train_data, n_future_train, n_past):

        x_train = []
        y_train = []

        for i in range(n_past, len(train_data)):
            x_train.append(train_data[i-n_past:i, :])       # uses n_past num variables for predictions, basically sliding window
            y_train.append(train_data[i + n_future_train - 1, :]) # currently predicting one day ahead

        # Convert the x_train and y_train to numpy arrays 
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        print('x_train shape == {}.' .format(x_train.shape))
        print('y_train shape == {}.' .format(y_train.shape))

        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(LSTM(16, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(y_train.shape[1]))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                                mode ="min", patience = 5, 
                                                restore_best_weights = True)
        # Train the model
        self.model.fit(x_train, y_train, batch_size=128, epochs=40, callbacks=[earlystopping])

        return x_train

    def test_lstm(self):
        dataset = self.close_data.values

        # Get the number of rows to train the model on
        training_data_len = int(np.ceil( len(dataset) * .95 ))

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Create the scaled training data set
        train_data = scaled_data[0:int(training_data_len), :]

        n_future_train = 1
        n_past = 2

        # train the data
        self.train(train_data, n_future_train, n_past)

        # Create the testing data set
        test_data = scaled_data[training_data_len - n_past: , :]

        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(n_past, len(test_data)):
            x_test.append(test_data[i-n_past:i, :])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2] ))

        # Get the models predicted price values 
        predictions = self.model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        train = self.close_data[:training_data_len]
        valid = self.close_data[training_data_len:]
        valid['Predictions'] = predictions

        correlation_matrix = np.corrcoef(valid['Close'], valid['Predictions'])
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2

        print('R^2 fit == {}.' .format(r_squared))

        # Plot the test data
        train_graph = go.Scatter(x=self.past.iloc[:training_data_len, 0], y=train['Close'],
                    mode='lines', name='Training',
                    opacity=0.8, marker_color='blue')
        pred_graph = go.Scatter(x=self.past.iloc[training_data_len:,0], y=valid['Predictions'],
                            mode='lines', name='Predictiion',
                            opacity=0.8, marker_color='red')
        valid_graph = go.Scatter(x=self.past.iloc[training_data_len:,0], y=valid['Close'],
                            mode='lines', name='Valid',
                            opacity=0.8, marker_color='green')
        self.test_fig = go.Figure()
        self.test_fig.add_trace(train_graph)
        self.test_fig.add_trace(pred_graph)
        self.test_fig.add_trace(valid_graph)
        self.test_fig.update_layout(
            title="Training the LSTM NN for " + self.ticker,
            yaxis_title="Closing Price ($)",
            xaxis_title="Date",
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgb(0,0,0,0)", 
            autosize=True,
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            margin=dict(
                l=30,
                r=30,
                b=60,
                t=50,
                pad=4
            ),
        )

        return r_squared
    
    def predict(self):
        n_future = 10
        forecast_period = pd.date_range(list(self.past['Date'])[-1] + pd.DateOffset(1), periods=n_future, freq='1d').to_series()

        dataset = self.close_data.values

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # retrain the model using the whole dataset
        train_data = scaled_data[:, :]

        n_future_train = 1
        n_past = 2

        # train the data
        x_train = self.train(train_data, n_future_train, n_past)

        predictions = self.model.predict(x_train[-n_future:])
        predictions = scaler.inverse_transform(predictions)
        self.predictions = pd.Series(predictions.flatten())

        # valid = self.close_data[-n_future:]
        # past_dates = self.past.iloc[-n_future:, 0]

        # valid_graph = go.Scatter(x=past_dates, y=valid['Close'],
        #             mode='lines', name='Prev 10 days',
        #             opacity=0.8, marker_color='red')

        pred_graph = go.Scatter(x=forecast_period, y=self.predictions,
                    mode='lines', name='Predicted',
                    opacity=0.8, marker_color='green')

        self.predict_fig = go.Figure()
        self.predict_fig.add_trace(pred_graph)
        # self.predict_fig.add_trace(valid_graph)
        self.predict_fig.update_layout(
            title="Prediction for the next 10 days: " + self.ticker,
            yaxis_title="Closing Price ($)",
            xaxis_title="Date",
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgb(0,0,0,0)", 
            autosize=True,
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            margin=dict(
                l=30,
                r=30,
                b=60,
                t=50,
                pad=4
            ),
        )
    
    # get the div for the figure
    def get_div(self, fig):

        plot_div = plot(fig,
               output_type='div')
        return plot_div
    
    def calc_macd(self):

        # Find the MACD difference for the closing price
        self.past['MACD_diff'] = ta.trend.macd_diff(self.past.Close)

        # Set the conditions and choices
        conditions = [
            (self.past.MACD_diff > 0) & (self.past.MACD_diff.shift(1) < 0),
            (self.past.MACD_diff < 0) & (self.past.MACD_diff.shift(1) > 0)]
        choices = [BUY, SELL]

        # Set the decision for MACD
        self.past['Decision_MACD'] = np.select(conditions, choices, default=STAY)

    def calc_rsi_sma(self):

        # Find sma200 and RSI
        self.past['sma200'] = ta.trend.sma_indicator(self.past.Close, window=200)
        self.past['RSI'] = ta.momentum.rsi(self.past.Close, window=10)

        # Set the conditions and choices
        conditions = [
            (self.past.Close > self.past.sma200) & (self.past.RSI < 30),
            ((self.past.Close > self.past.sma200) & (self.past.RSI < 30)) | ((self.past.Close.shift(10) > self.past.sma200.shift(10)) & (self.past.RSI.shift(10) < 30))]
        choices = [BUY, SELL]

        # Set the decision for the RSI/SMA strategy
        self.past['Decision_RSI_SMA'] = np.select(conditions, choices, default=STAY)

    def calc_candle(self):

        # retrieve sma50, body high, body low, body, body avg
        self.past['sma50'] = ta.trend.sma_indicator(self.past.Close, window=50)
        self.past['Body_Hi'] = self.past[['Open', 'Close']].max(axis=1)
        self.past['Body_Lo'] = self.past[['Open', 'Close']].min(axis=1)
        self.past['Body'] = self.past.Body_Hi - self.past.Body_Lo
        self.past['Body_Avg'] = ta.trend.ema_indicator(self.past.Body, window=PREV_DEPTH_BODY_AVG)

        self.past['Decision_candle'] = self.search_candle_patterns(self.past)

    # get the decision for buy and sell concerning candlesticks
    def search_candle_patterns(self, df):

        # make list to hold buy, sell, stay decisions
        decision_cand = []

        # iterate through given dataframe
        for idx in range(len(df)):

            # check if the sma50 value is Nan
            if isnan(df['sma50'].iloc[idx]):
                decision_cand.append(STAY)
            # check if 30 spaces behind is a buy signal, then set to sell
            elif decision_cand[-30] == BUY:
                decision_cand.append(SELL)
            # look for a candlestick pattern, if no pattern is found then 
            else:
                downtrend = df['Close'].iloc[idx] < df['sma50'].iloc[idx]
                uptrend = df['Close'].iloc[idx] < df['sma50'].iloc[idx]
                is_match = False

                if downtrend:
                    is_match = self.bullish_engulfing(idx, df)
                if uptrend:
                    is_match = self.bearish_engulfing(idx, df)
                
                if is_match:
                    decision_cand.append(BUY)
                else:
                    decision_cand.append(STAY)

        # return decision_cand so it can be turned into a column in self.past
        return decision_cand

    # checks if candlestick matches the engulfing bullish pattern
    def bullish_engulfing(self, i, data):

        white_body = data['Open'].iloc[i] < data['Close'].iloc[i]
        long_body = data['Body'].iloc[i]  > data['Body_Avg'].iloc[i]
        prev_black_body = data['Open'].iloc[i-1] > data['Close'].iloc[i-1]
        prev_small_body = data['Body'].iloc[i-1]  < data['Body_Avg'].iloc[i-1]

        if (white_body and long_body and prev_black_body and prev_small_body 
                and data['Close'].iloc[i] >= data['Open'].iloc[i-1] and data['Open'].iloc[i] <= data['Close'].iloc[i-1]
                and (data['Close'].iloc[i] > data['Open'].iloc[i-1] or data['Open'].iloc[i] < data['Close'].iloc[i-1])):
            return True

        return False

    # checks if candlestick matches the engulfing bearish pattern
    def bearish_engulfing(self, i, data):
        
        black_body = data['Open'].iloc[i] > data['Close'].iloc[i]
        long_body = data['Body'].iloc[i] > data['Body_Avg'].iloc[i]

        prev_white_body = data['Open'].iloc[i-1] < data['Close'].iloc[i-1]
        prev_small_body = data['Body'].iloc[i-1] < data['Body_Avg'].iloc[i-1]

        if (black_body and long_body and prev_white_body and prev_small_body
                and data['Close'].iloc[i] <= data['Open'].iloc[i-1] and
                data['Open'].iloc[i] >= data['Close'].iloc[i-1] and
                (data['Close'].iloc[i] < data['Open'].iloc[i-1] or
                data['Open'].iloc[i] > data['Close'].iloc[i-1])):
            return True

        return False
    
    # return decision based on the prediction
    def prediction_decision(self):
        
        decision_p = STAY

        # check if the lst item is the smallest or largest price
        if self.predictions.index[-1] == self.predictions.idxmax():
            decision_p = SELL
        elif self.predictions.index[-1] == self.predictions.idxmin():
            decision_p = BUY

        return decision_p
    
    # recommendation function that returns strings for each type of strategy
    def recommendation(self):

        r_squared = self.test_lstm()
        self.predict()
        self.calc_macd()
        self.calc_rsi_sma()
        self.calc_candle()

        MACD_list = []
        RSI_list = []
        Candle_list = []
        predict_list = []

        MACD_list.append("Based on the MACD strategy: " + self.decision_to_str(self.past['Decision_MACD'].iloc[-1]))
        MACD_win, MACD_profit = self.backtest('Decision_MACD')
        MACD_list.append("MACD average profits: " + str("{:.2f}".format(MACD_profit)) + "%")
        MACD_list.append("MACD win rate: " + str("{:.2f}".format(MACD_win)) + "%")

        RSI_list.append("Based on the RSI strategy: " + self.decision_to_str(self.past['Decision_RSI_SMA'].iloc[-1]))
        RSI_win, RSI_profit = self.backtest('Decision_RSI_SMA')
        RSI_list.append("RSI average profit: " + str("{:.2f}".format(RSI_profit)) + "%")
        RSI_list.append("RSI win rate: " + str("{:.2f}".format(RSI_win)) + "%")

        Candle_list.append("Based on the Candlestick Pattern strategy: " + self.decision_to_str(self.past['Decision_candle'].iloc[-1]))
        Candle_win, Candle_profit = self.backtest('Decision_candle')
        Candle_list.append("Candlestick Pattern average profit: " + str("{:.2f}".format(Candle_profit)) + "%")
        Candle_list.append("Candlestick Pattern win rate: " + str("{:.2f}".format(Candle_win)) + "%")

        predict_list.append("Based on our prediction results: " + self.decision_to_str(self.prediction_decision()))
        predict_list.append("R\u00b2 goodness of fit for our prediction: " + str("{:.2f}".format(r_squared)))

        recommendation_list = {
            'MACD': MACD_list,
            'RSI': RSI_list,
            'Candle': Candle_list,
            'Predict': predict_list
        }

        return recommendation_list
    
    # convert decision number to a string
    def decision_to_str(self, decision):

        dec_string = ""

        if decision == BUY:
            dec_string = "BUY"
        elif decision == SELL:
            dec_string = "SELL"
        else:
            dec_string = "STAY"
        
        return dec_string

    # backtest function to see profit based on strategy:
    def backtest(self, dec_str):

        if dec_str in self.past.columns:

            buy_indices = self.past.index[self.past[dec_str] == BUY].tolist()
            buy_indices = [i+1 for i in buy_indices]
            sell_indices = self.past.index[self.past[dec_str] == SELL].tolist()
            sell_indices = [i+1 for i in sell_indices]

            # make sure that buying date comes first
            if sell_indices[0] < buy_indices[0]:
                sell_indices.pop(0)
            elif buy_indices[-1] > sell_indices[-1]:
                buy_indices.pop()

            # make sure that indices don't go out of bounds
            if sell_indices[-1] >= len(self.past.Open):
                sell_indices.pop()
            if buy_indices[-1] >= len(self.past.Open):
                buy_indices.pop()
            
            buy_prices = self.past.Open.iloc[buy_indices]
            sell_prices = self.past.Open.iloc[sell_indices]

            profits = []

            for i in range(len(sell_prices)):
                profits.append((sell_prices.iloc[i] - buy_prices.iloc[i])/buy_prices.iloc[i])
            
            wins = [i for i in profits if i > 0]

            percent_wins = (len(wins)/len(profits)) * 100
            avg_profit = (sum(profits)/len(profits)) * 100

            return percent_wins, avg_profit
        
        return

def main():
    
    # resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # soup = bs.BeautifulSoup(resp.text, 'lxml')
    # table = soup.find('table', {'class': 'wikitable sortable'})
    # tickers = []
    # for row in table.findAll('tr')[1:]:
    #     ticker = row.findAll('td')[0].text
    #     tickers.append(ticker)
    # tickers = list(map(lambda s: s.strip(), tickers))
    # # print(tickers)

    # # tickers = tickers[]

    # data = []
    # i = 1

    # for ticker in tickers:
    #     try:
    #         print(str(i) + " " + ticker)
    #         predict_stock = predictor(ticker)
    #         predict_stock.recommendation()
    #         MACD_wins, MACD_profit = predict_stock.backtest('Decision_MACD')
    #         RSI_wins, RSI_profit = predict_stock.backtest('Decision_RSI_SMA')
    #         Candle_wins, Candle_profit = predict_stock.backtest('Decision_candle')
    #         data.append({'Ticker': ticker, 'MACD win rate': MACD_wins, 'MACD avg profit': MACD_profit, 'RSI win rate': RSI_wins, 'RSI avg profit': RSI_profit, 'Candle win rate': Candle_wins, 'Candle avg profit': Candle_profit})
    #     except:
    #         print(ticker + "not available")
    #         continue
        

    # final_data = pd.DataFrame(data)
    # final_data.to_csv('Strategy_comparison.csv')


    predict_aapl = predictor('aapl')
    predict_aapl.test_lstm()
    predict_aapl.predict()
    # predict_aapl.test_fig.show()
    predict_aapl.predict_fig.show()
    print(predict_aapl.predictions.index[-1] == predict_aapl.predictions.idxmax())
    print(predict_aapl.predictions.index[-1] == predict_aapl.predictions.idxmin())


    # predict_aapl.calc_macd()
    # predict_aapl.calc_rsi_sma()
    # predict_aapl.calc_candle()
    # print(predict_aapl.past.Decision_MACD)
    # predict_aapl.past.to_csv("macd.csv", columns=['Decision_MACD'])
    # predict_aapl.past.to_csv("rsisma.csv", columns=['Decision_RSI_SMA'])
    # predict_aapl.past.to_csv("candle.csv", columns=['Decision_candle'])
    # print(predict_aapl.past.sma200.head(5))
    # print(predict_aapl.recommendation())
    # MACD_wins, MACD_profit = predict_aapl.backtest('Decision_MACD')
    # RSI_wins, RSI_profit = predict_aapl.backtest('Decision_RSI_SMA')
    # Candle_wins, Candle_profit = predict_aapl.backtest('Decision_candle')
    # print("MACD avg profit: " + str("{:.2f}".format(MACD_profit)) + "%")
    # print("MACD wins: " + str("{:.2f}".format(MACD_wins)) + "%")
    
    # print("MACD avg profit: " + str("{:.2f}".format(RSI_profit)) + "%")
    # print("MACD wins: " + str("{:.2f}".format(RSI_wins)) + "%")
    
    # print("MACD avg profit: " + str("{:.2f}".format(Candle_profit)) + "%")
    # print("MACD wins: " + str("{:.2f}".format(Candle_wins)) + "%")
    


if __name__ == '__main__':
    main()
    

