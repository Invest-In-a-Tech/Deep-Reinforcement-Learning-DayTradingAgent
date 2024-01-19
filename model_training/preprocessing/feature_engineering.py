import pandas as pd


class FeatureEngineering:
    def __init__(self, df):
        # Copying DataFrame if necessary
        self.df = df if df is not None else pd.DataFrame()


    def perform_feature_engineering(self):
        """
        Enhancements for Performance:
        
        - Used vectorized group operations with pandas groupby and apply methods. 
          This avoids looping over groups manually and leverages pandas' optimized 
          internal computations for group operations.
        
        - Custom function 'calculate_levels' is applied to each group to perform 
          necessary calculations in a vectorized way, increasing efficiency.
        
        - Efficient merging with the original DataFrame and in-place handling of 
          missing values further enhances performance.
        """       
        # Ensure DataFrame has necessary columns
        required_columns = {'TotalVolume', 'Price'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Group data by Date
        if not self.df.index.is_unique:
            grouped = self.df.groupby(self.df.index)
        else:
            # If each row has a unique index, grouping is not necessary
            grouped = [(self.df.index[i], self.df.iloc[[i]]) for i in range(len(self.df))]


        def calculate_levels(group):
            sorted_volumes = group.sort_values('TotalVolume')
            poc = sorted_volumes.iloc[-1]
            hvn = sorted_volumes.iloc[-2] if len(sorted_volumes) > 1 else poc
            lvn = sorted_volumes.iloc[0]

            return pd.Series({
                'POC_Price': poc['Price'],
                'POC_Volume': poc['TotalVolume'],
                'HVN_Price': hvn['Price'],
                'HVN_Volume': hvn['TotalVolume'],
                'LVN_Price': lvn['Price'],
                'LVN_Volume': lvn['TotalVolume']
            })

        combined_df = grouped.apply(calculate_levels)

        # Merge with Original DataFrame
        self.df = self.df.merge(combined_df, left_index=True, right_index=True, how='left')


        # Handle Missing Values
        self.df.ffill(inplace=True)

        #print(self.df.tail(50))
        print(f"After processing footprint dataframe, DataFrame shape: {self.df.shape}")
        return self.df.dropna()