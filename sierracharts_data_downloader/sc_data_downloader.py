
# Import libraries
import pandas as pd
from trade29.sc.bridge import SCBridge
import os
from dotenv import find_dotenv, load_dotenv


# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


# Process the DataFrame
def process_dataframe(df):
    df = df.sort_values(by=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Date', inplace=True)
    return df


# Create an instance of the SCBridge class
sc = SCBridge()


# Request the data
sc.graph_data_request(
    sc_api_key, 
    historical_init_bars=5, # Max historical bars is 600,000
    include_vbp=True, 
    on_bar_close=True, 
    base_data='1;2;3;4;5', 
    sg_data="ID2.[SG1;SG10];ID4.[SG1];ID5.[SG2-SG3];ID6.[SG1]"
)


# Wait for the response
msg = sc.get_response_queue().get()


# Rename columns for the entire msg.df
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


# Process tuple data using vectorized operations
exploded_df = msg.df.explode('Tuples')
tuple_cols = ['Price', 'Bid', 'Ask', 'TotalVolume', 'NumberOfTrades']
exploded_df[tuple_cols] = pd.DataFrame(exploded_df['Tuples'].tolist(), index=exploded_df.index)
exploded_df.drop(columns=['BarNumber', 'Tuples'], inplace=True)
vbp_df = process_dataframe(exploded_df)
print(vbp_df.shape)
#print(vbp_df.tail(10))


# Save the DataFrame to a CSV file
file_path = os.path.join('model_training', 'data', 'ES_tuples.csv')
vbp_df.to_csv(file_path, index=True) 
print(f"DataFrame saved to {file_path}")