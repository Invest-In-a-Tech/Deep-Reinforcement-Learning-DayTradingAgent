# footprint_dataframe.py

import pandas as pd

class FootprintDataframe:
    def __init__(self, df):
        # Copying DataFrame if necessary
        self.df = df if df is not None else pd.DataFrame()


    def process_footprint_dataframe(self):
        # Ensure DataFrame has necessary columns
        required_columns = {'TotalVolume', 'Price'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Group data by Date if the index is meaningful (e.g., dates)
        if not self.df.index.is_unique:
            grouped = self.df.groupby(self.df.index)
        else:
            # If index is unique for each row, the grouping is not necessary
            grouped = [(self.df.index[i], self.df.iloc[[i]]) for i in range(len(self.df))]

        combined_data = []

        for date, group in grouped:
            # Simplify calculations
            sorted_volumes = group['TotalVolume'].sort_values()
            max_volume = sorted_volumes.iloc[-1]
            poc_price_level = group[group['TotalVolume'] == max_volume]['Price'].iloc[0]

            hvn_volume = sorted_volumes.iloc[-2] if len(sorted_volumes) > 1 else max_volume
            hvn_price_level = group[group['TotalVolume'] == hvn_volume]['Price'].iloc[0]

            lvn_volume = sorted_volumes.iloc[0]
            lvn_price_level = group[group['TotalVolume'] == lvn_volume]['Price'].iloc[0]

            combined_data.append((date, poc_price_level, max_volume, hvn_price_level, hvn_volume, lvn_price_level, lvn_volume))

        combined_df = pd.DataFrame(combined_data, columns=['Date', 'POC_Price', 'POC_Volume', 'HVN_Price', 'HVN_Volume', 'LVN_Price', 'LVN_Volume'])
        combined_df.set_index('Date', inplace=True)

        # Merge with Original DataFrame
        self.df = self.df.merge(combined_df, left_index=True, right_index=True, how='left')

        # Handle Missing Values
        self.df.ffill()

        # Select and print the specified columns
        columns_to_print = ['Open', 'High', 'Low', 'Close', 'Volume', 'Delta', 'CVD', 'Price', 'Bid', 'Ask', 'TotalVolume', 
                            'POC_Price', 'HVN_Price', 'LVN_Price', 'POC_Volume', 'HVN_Volume', 'LVN_Volume']
        
        print(self.df.tail(10))
        print(f"After processing footprint dataframe, DataFrame shape: {self.df.shape}")
        return self.df.dropna()
    
