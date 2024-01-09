# In this file, we are subscribing to tuples data and processing it into a DataFrame

import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv


# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


class DataEvent:
    def __init__(self, sc):
        self.sc = sc
        self.subscribe()


    def subscribe(self):
        self.data_reqid = self.sc.graph_data_request(
            key=sc_api_key, 
            historical_init_bars=30, 
            realtime_update_bars=50, 
            include_vbp=True, 
            #update_frequency=1,
            on_bar_close=True, 
            base_data='1;2;3;4;5', 
            sg_data="ID2.[SG1;SG10];ID4.[SG1];ID5.[SG2-SG3];ID6.[SG1]",
        )


    def process_dataframe(self, df):
        df = df.sort_values(by=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('Date', inplace=True)
        df = df.dropna()  # Remove rows with missing values
        return df


    def process_data_event(self, msg):
        msg.df.columns = [
            'Date', 
            'BarNumber', 
            'Open', 
            'High', 
            'Low', 
            'Close', 
            'Volume', 
            'Delta', 
            'CVD', 
            'TodayOpen', 
            'PrevHigh',
            'PrevLow', 
            'VWAP', 
            'Tuples'
        ]


        exploded_df = msg.df.explode('Tuples')
        tuple_cols = ['Price', 'Bid', 'Ask', 'TotalVolume', 'NumberOfTrades']
        exploded_df[tuple_cols] = pd.DataFrame(exploded_df['Tuples'].tolist(), index=exploded_df.index)
        exploded_df.drop(columns=['BarNumber', 'Tuples'], inplace=True)
        print(f"After processing data, DataFrame shape: {exploded_df.shape}")
        return self.process_dataframe(exploded_df)

