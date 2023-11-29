"""
@author: Gautam Bamba
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from ta.trend import ADXIndicator #For this library, run "pip install ta" command



#The folderpaths for the train and test data
train_path = "/Users/gautammacbook/Downloads/Nomura Assn/StockData_Train"
test_path = "/Users/gautammacbook/Downloads/Nomura Assn/StockData_Test"

#Specifies the number of trading days. There are 18 holidays in the SSE except weekends
N_trading = 242

### Setting the working directory to the folder containing the stock data (train/test data)
os.chdir(train_path)

### Getting the list of stock data filepaths
files = os.listdir(train_path)
files.sort()
del files[0]

### Creating an empty dataframe to store the entire time-series data of all 500 stocks
data = pd.DataFrame(columns=["Date", "Time", "Symbol", "Open", "High", "Low", "Close", "Volume", "Amount"])

#Looping through the files to extract stock data and adding it to the dataframe
for file in files:
    temp = pd.read_csv(file, usecols=["date", "time", "code", "open", "high", "low", "close", "volume", "amount"])
    temp.rename(columns={"date":"Date", "time": "Time", "code":"Symbol", "open":"Open", "high":"High", 
                         "low":"Low", "close":"Close", "volume":"Volume", "amount":"Amount"}, inplace=True)
    temp["Time"] = [str(x)[:-5] for x in temp["Time"]]
    data = pd.concat([data, temp])
data.reset_index(drop=True, inplace=True)

### Get the returns for each stock at every timestamp
data["Returns"] = data.groupby(["Symbol"], sort=False)["Close"].pct_change().fillna(0.0)
#==========================================================================================================


#==========================================================================================================
                                        ###### Buy and Hold ######
# We first find out the stocks that were active (not suspended from trading) at each timestamp.
active_stocks = data.groupby(["Time"])["Close"].count().to_frame()
active_stocks.rename(columns={"Close":"Active Stocks"}, inplace=True)
active_stocks.reset_index(inplace=True)
data = data.merge(active_stocks, on="Time", how="left")
data_BnH = data[["Date", "Time", "Symbol", "Close", "Returns", "Active Stocks"]]

# To have equally-weighted stocks, at each timestamp, the signal for every active stock is 100/(No. of active stocks that day)
data_BnH["Signal"] = 100 / data_BnH["Active Stocks"]

#For each stock, PnL_t = Signal_(t-1) * Returns_t
data_BnH["Signal_(t-1)"] = data_BnH.groupby(["Symbol"], sort=False)["Signal"].shift(periods=1, fill_value=0)
data_BnH["PnL"] = data_BnH["Signal_(t-1)"] * data_BnH["Returns"]
data_BnH["Value"] = data_BnH["Close"] * data_BnH["Signal"]

#This dataframe stores the total portfolio's PnLs at each timestamp
PnL_BnH = data_BnH.groupby(["Time"], sort=False)["PnL"].sum().to_frame().reset_index()
x2 = data_BnH.groupby(["Time"], sort=False)["Value"].sum().to_frame().reset_index()
PnL_BnH = PnL_BnH.merge(x2, on="Time", how="left")
PnL_BnH["Cumu_PnL"] = PnL_BnH["PnL"].cumsum()

#The annual Sharpe ratio of the Buy and Hold strategy
Sharpe_BnH = math.sqrt(8*N_trading)*np.mean(PnL_BnH["PnL"])/np.std(PnL_BnH["PnL"])

#The annual Sortino ratio of the Buy and Hold strategy
Sortino_BnH = math.sqrt(8*N_trading) * PnL_BnH["PnL"].mean() / PnL_BnH["PnL"][PnL_BnH["PnL"]<0].std()
#==========================================================================================================



#==========================================================================================================
                                        ### Our Strategy ###
# We have taken Bollinger Bands, RSI, MACD and ADX as features for the strategy

#==========================================================================================================
### Bollinger Bands: We calculate 20-period moving avg and moving std dev. to calculate BBs for each stock
### If our Z-score, i.e. (Close Price - SMA20) / Mov. StdDev., is less than -2, the asset is overextended 
### and is likely to revert to its mean
### Signal - {-1,0,1}
Period_Boll = 20
Boll_exit = 0
data["SMA_20"] = data.groupby(["Symbol"], sort=False)["Close"].rolling(Period_Boll).mean().reset_index(0,drop=True)
data["Std_Dev"] = data.groupby(["Symbol"], sort=False)["Close"].rolling(Period_Boll).std().reset_index(0,drop=True)
data["Z-score"] = (data["Close"] - data["SMA_20"]) / data["Std_Dev"]

### We buy when price hits Lower BB, and square-off when price reaches moving average. 
### We sell when price hits Upper BB, and square-off when price reaches moving average.
data["Signal_Boll"] = 0
data.loc[data["Z-score"] < -2, "Signal_Boll"] = 1
data.loc[data["Z-score"] > 2, "Signal_Boll"] = -1

grouped3 = data.groupby(["Symbol"], sort=False)
def Boll_Signal(df):
    #df.reset_index(0, drop=True)
    for i in df.index[1:]:
        if(df.loc[i,"Signal_Boll"]==0):
            if((df.loc[i-1,"Signal_Boll"]==1) & (df.loc[i, "Z-score"]<-Boll_exit)):
                df.loc[i, "Signal_Boll"] = 1
            
            if((df.loc[i-1, "Signal_Boll"]==-1) & (df.loc[i, "Z-score"]>Boll_exit)):
                df.loc[i, "Signal_Boll"] = -1
            
    return df
data = grouped3.apply(Boll_Signal)
data = data.droplevel(0)
#==========================================================================================================


#==========================================================================================================
### RSI: Lies between 0 to 100, gives us a measure of how overbought or oversold an asset is.
### Signal - {-1,0,1}
Period_RSI = 14
data["gain"] = data["Returns"].clip(lower=0)
data["loss"] = data["Returns"].clip(upper=0).abs()
data["Avg_up"] = data.groupby(["Symbol"], sort=False)["gain"].ewm(com = Period_RSI-1, adjust=True).mean().reset_index(0,drop=True)
data["Avg_down"] = data.groupby(["Symbol"], sort=False)["loss"].ewm(com = Period_RSI-1, adjust=True).mean().reset_index(0,drop=True)
data["RSI"] = 100 * data["Avg_up"] / (data["Avg_up"] + data["Avg_down"])

### We buy when RSI goes below 30, and square-off when RSI reaches back to 50 
### We buy when RSI goes above 70, and square-off when RSI reaches back to 50 
data["Signal_RSI"] = 0
data.loc[data["RSI"] > 70, "Signal_RSI"] = -1
data.loc[data["RSI"] < 30, "Signal_RSI"] = 1

grouped4 = data.groupby(["Symbol"], sort=False)
def RSI_Signal(df):
    for i in df.index[1:]:
        if(df.loc[i,"Signal_RSI"]==0):
            if((df.loc[i-1,"Signal_RSI"]==1) & (df.loc[i, "RSI"]<50)):
                df.loc[i, "Signal_RSI"] = 1
            
            if((df.loc[i-1, "Signal_RSI"]==-1) & (df.loc[i, "RSI"]>50)):
                df.loc[i, "Signal_RSI"] = -1
    return df
data = grouped4.apply(RSI_Signal)
data = data.droplevel(0)
#==========================================================================================================


#==========================================================================================================
### MACD(Moving Average Convergance/Divergence): MACD Line indicates bullish trend when it is +ve, bearish when -ve
### Generates a Buy signal when the MACD Line moves above the MACD Signal line (9-period EMA of MACD Line)
### Signal - {0,1}
data["EMA_12"] = data.groupby(["Symbol"], sort=False)["Close"].ewm(span=12, adjust=True).mean().reset_index(0,drop=True)
data["EMA_26"] = data.groupby(["Symbol"], sort=False)["Close"].ewm(span=26, adjust=True).mean().reset_index(0,drop=True)
data["MACD"] = data["EMA_12"] - data["EMA_26"]
data["MACD_Sig"] = data.groupby(["Symbol"], sort=False)["MACD"].ewm(span=9, adjust=True).mean().reset_index(0,drop=True)
data["MACD_Hist"] = data["MACD"] - data["MACD_Sig"]

data["Signal_MACD"] = np.sign(data["MACD_Hist"])
data["Signal_MACD"] = data["Signal_MACD"].clip(lower=0)
#==========================================================================================================


#==========================================================================================================
### ADX(Average Directional Index): Gives us a measure of the strength of an ongoing trend
### An ADX below 25 means a very weak trend, so reversion might be imminent
### Signal - {0,1}
grouped = data.groupby("Symbol", sort=False)
def ADX_Ind(df):
    adxI = ADXIndicator(df['High'], df['Low'], df['Close'], 14, True)
    df["ADX"] = adxI.adx()
    return df

data = grouped.apply(ADX_Ind)
data = data.droplevel(0)
data["Signal_ADX"] = 0
data.loc[data["ADX"] < 25, "Signal_ADX"] = 1
data.loc[data["ADX"] == 0, "Signal_ADX"] = 0
#==========================================================================================================


#==========================================================================================================
### Combining signals from the features 
### Strategy Signal = Sum(Feature Signals)
data["Signal_Sum"] = data["Signal_Boll"] + data["Signal_RSI"] + data["Signal_ADX"] + data["Signal_MACD"]
data["Signal_Strategy"] = data["Signal_Sum"].clip(lower=0)

#If RSI or Bollinger band signal is -1, the Strategy Signal is taken 0 (Likely reversion in downward direction)
data.loc[(data["Signal_Boll"]==-1) | (data["Signal_RSI"]==-1), "Signal_Strategy"] = 0
#If only 1 of the 4 indicators is indicating a Buy signal, the Strategy Signal is taken 0 (Likely a false positive)
data.loc[data["Signal_Sum"]==1, "Signal_Strategy"] = 0
#==========================================================================================================


#==========================================================================================================
### Final Signal Determination
# Calculate the sum of signals over all stocks for each timestamp
# We scale down each individual signal of each stocks to ensure a total position of 100 for each timestamp
total_signal = data.groupby(["Time"], sort=False)["Signal_Strategy"].sum().to_frame()
total_signal.rename(columns={"Signal_Strategy":"Total_Signal"}, inplace=True)
total_signal.reset_index(inplace=True)

data = data.merge(total_signal, on="Time", how="left")
data["Signal_Final"] = data["Signal_Strategy"] * 100 / data["Total_Signal"]
#For timestamps where total signal = 0, the default holding is taken as equally-weighted over each active stock 
data.loc[data["Total_Signal"]==0, "Signal_Final"] = 100 / data["Active Stocks"]
#==========================================================================================================


#==========================================================================================================
### PnL Calculation
#Calculates PnL for each stock at each timestamp
data["Signal_(t-1)"] = data.groupby(["Symbol"], sort=False)["Signal_Final"].shift(periods=1, fill_value=0)
data["PnL"] = data["Signal_(t-1)"] * data["Returns"]
data["Value"] = data["Signal_Final"] * data["Close"]
# Calculates the portfolio's PnL for each timestamp
PnL_strat = data.groupby(["Time"], sort=False)["PnL"].sum().to_frame().reset_index()
x = data.groupby(["Time"], sort=False)["Value"].sum().to_frame().reset_index()
PnL_strat = PnL_strat.merge(x, on="Time", how="left")
PnL_strat["Cumu_PnL"] = PnL_strat["PnL"].cumsum()
    
#The annual Sharpe ratio of the strategy
Sharpe_strat = math.sqrt(8*N_trading)*np.mean(PnL_strat["PnL"])/np.std(PnL_strat["PnL"])

#The annual Sortino ratio of the strategy
Sortino_strat = math.sqrt(8*N_trading) * PnL_strat["PnL"].mean() / PnL_strat["PnL"][PnL_strat["PnL"]<0].std()
#==========================================================================================================

###Plotting graphs
plt.figure(1)
plt.plot(PnL_strat["Cumu_PnL"], label="Strategy", linewidth=1)
plt.plot(PnL_BnH["Cumu_PnL"], label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("Strategy vs. Buy and Hold - Out-Sample Data"); 
    