# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import baostock as bs
import pandas as pd
import numpy as np
import os

stockfolder_path = "/Users/gautammacbook/Downloads/Nomura Assn"
stockfile_path = "/Users/gautammacbook/Downloads/Nomura Assn/zz500_stocks.csv"
train_path = "/Users/gautammacbook/Downloads/Nomura Assn/StockData_Train"
test_path = "/Users/gautammacbook/Downloads/Nomura Assn/StockData_Test"

os.chdir(stockfolder_path)

# login system
lg = bs.login()
# Display login return information
print('login respond error_code:'+lg.error_code)
print('login respond error_msg:'+lg.error_msg)

# Get CSI 500 constituent stocks
rs = bs.query_zz500_stocks(date = "2021-01-01")
print('query_zz500 error_code:'+rs.error_code)
print('query_zz500 error_msg:'+rs.error_msg)

#Print the result set
zz500_stocks = []
while (rs.error_code == '0') & rs.next():
    # Get a record and merge the records together
    zz500_stocks.append(rs.get_row_data())
result = pd.DataFrame(zz500_stocks, columns=rs.fields)
# Output the result set to a csv file
result.to_csv("zz500_stocks.csv", encoding="gbk", index=False)

# Log out of the system
bs.logout()

stock_list = pd.read_csv(stockfile_path, encoding="gbk", usecols=["code"])


### Extracting in-sample data ###
os.chdir(train_path)

#### login system####
lg = bs.login()
# Display login return information
print('login respond error_code:'+lg.error_code)
print('login respond error_msg:'+lg.error_msg)

#### Obtain historical K-line data of Shanghai and Shenzhen A shares####
# For detailed indicator parameters, please refer to the "Historical Market Indicator Parameters" chapter; the "minute line" parameters are different from the "daily line" parameters. The "minute line" does not contain an index.
# Minute line indicators: date, time, code, open, high, low, close, volume, amount, adjustflag

for stock in stock_list:
    rs = bs.query_history_k_data_plus(stock,
        "date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2022-04-01', end_date='2022-06-30',
        frequency="30", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond error_msg:'+rs.error_msg)
    
    #### Print result set####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # Get a record and merge the records together
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    
    #### Output the result set to a csv file####   
    result.to_csv(stock + "_train.csv", index=False)
    #print(result)

#### Log out of the system####
bs.logout()



### Extracting out-sample data ###
os.chdir(test_path)

#### login system####
lg = bs.login()
# Display login return information
print('login respond error_code:'+lg.error_code)
print('login respond error_msg:'+lg.error_msg)

#### Obtain historical K-line data of Shanghai and Shenzhen A shares####
# For detailed indicator parameters, please refer to the "Historical Market Indicator Parameters" chapter; the "minute line" parameters are different from the "daily line" parameters. The "minute line" does not contain an index.
# Minute line indicators: date, time, code, open, high, low, close, volume, amount, adjustflag

for stock in stock_list:
    rs = bs.query_history_k_data_plus(stock,
        "date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2022-04-01', end_date='2022-06-30',
        frequency="30", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond error_msg:'+rs.error_msg)
    
    #### Print result set####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # Get a record and merge the records together
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    
    #### Output the result set to a csv file####   
    result.to_csv(stock + "_train.csv", index=False)
    #print(result)

#### Log out of the system####
bs.logout()

