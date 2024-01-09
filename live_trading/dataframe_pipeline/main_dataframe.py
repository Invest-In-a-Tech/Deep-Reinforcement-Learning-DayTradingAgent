import pandas as pd
from live_trading.dataframe_pipeline.footprint_dataframe import FootprintDataframe

class MainDataframe:
    def __init__(self, df):
        # Copying DataFrame if necessary
        self.df = df if df is not None else pd.DataFrame()


    def process_main_dataframe(self):
        # Create an instance of FootprintDataframe
        footprintdataframe = FootprintDataframe(self.df)

        # Perform operations defined in FootprintDataframe
        fp_df = footprintdataframe.process_footprint_dataframe()

        # Grouping the data by timestamp
        grouped_data = fp_df.groupby(fp_df.index)

        # Preparing a DataFrame to hold the POC, HVN, and LVN for each timestamp
        poc_hvn_lvn_data = []

        for date, group in grouped_data:
            # Extracting POC, HVN, and LVN data for each timestamp
            # POC, HVN, and LVN data
            poc_price = group['POC_Price'].iloc[0]  # Assuming the first occurrence is the most relevant
            poc_volume = group['POC_Volume'].iloc[0]
            hvn_price = group['HVN_Price'].iloc[0]
            hvn_volume = group['HVN_Volume'].iloc[0]
            lvn_price = group['LVN_Price'].iloc[0]
            lvn_volume = group['LVN_Volume'].iloc[0]
            
            # Additional columns data
            open_price = group['Open'].iloc[0]
            high_price = group['High'].iloc[0]
            low_price = group['Low'].iloc[0]
            close_price = group['Close'].iloc[0]
            volume = group['Volume'].iloc[0]
            delta = group['Delta'].iloc[0]
            cvd = group['CVD'].iloc[0]

            poc_hvn_lvn_data.append((date,  poc_price, poc_volume, hvn_price, hvn_volume, lvn_price, lvn_volume, 
                          open_price, high_price, low_price, close_price, volume, delta, cvd))

        # Creating a DataFrame from the aggregated data
        columns = ['Date', 'POC_Price', 'POC_Volume', 'HVN_Price', 'HVN_Volume', 'LVN_Price', 'LVN_Volume', 
           'Open', 'High', 'Low', 'Close', 'Volume', 'Delta', 'CVD']
        
        new_dataframe = pd.DataFrame(poc_hvn_lvn_data, columns=columns)
        new_dataframe.set_index('Date', inplace=True)

        print(f"After processing main dataframe, DataFrame shape: {new_dataframe.shape}")
        print(new_dataframe.tail(10))
        # Return the enhanced DataFrame
        return new_dataframe
    
