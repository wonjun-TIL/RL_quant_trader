from pandas_datareader import data
import yfinance
yfinance.pdr_override()

chart_data = data.get_data_yahoo('005930.KS', '2023-09-01', '2023-09-13')

print(chart_data)
