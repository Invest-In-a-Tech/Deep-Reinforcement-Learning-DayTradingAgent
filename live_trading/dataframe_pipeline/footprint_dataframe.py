##############################################
# IMPORTS 
# ============================================
import pandas as pd




##############################################
# FeatureEngineering Class
# ============================================
class FeatureEngineering:
    """
    This class is designed for conducting feature engineering on financial market data. It processes a DataFrame 
    to calculate and append new features based on trading volume and price information, such as Point of Control (POC), 
    High Volume Node (HVN), and Low Volume Node (LVN) levels.

    The class implements efficient data processing techniques, including vectorized operations and pandas groupby 
    mechanisms, to enhance performance and ensure scalability for large datasets.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing financial market data. Expected to have 'TotalVolume' 
        and 'Price' columns among others.
    """
    def __init__(self, df):
        """
        Initializes the FeatureEngineering class with a DataFrame containing financial market data.

        Parameters:
            df (pd.DataFrame): The DataFrame to be processed. If None, an empty DataFrame is used.
        """
        self.df = df if df is not None else pd.DataFrame() # Copying DataFrame if necessary




    ########################################
    # Perform Feature Engineering
    # ======================================
    def perform_feature_engineering(self):
        """
        Processes the input DataFrame to calculate and append features 
        related to trading volume and price levels:Point of Control (POC),
        High Volume Node (HVN), and Low Volume Node (LVN).

        This method applies a custom function to efficiently compute these 
        levels for each group of data (typically grouped by date or time),
        and merges the results with the original DataFrame. Missing values 
        are forward-filled to maintain data continuity.
        
        Enhancements for Performance:
        
        - Used vectorized group operations with pandas groupby and apply methods. 
          This avoids looping over groups manually and leverages pandas' optimized 
          internal computations for group operations.
        
        - Custom function 'calculate_levels' is applied to each group to perform 
          necessary calculations in a vectorized way, increasing efficiency.
        
        - Efficient merging with the original DataFrame and in-place handling of 
          missing values further enhances performance.
          
        Returns:
            pd.DataFrame: The enhanced DataFrame with new volume and price level features appended.
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




        ########################################
        # Calculate Levels
        # ======================================
        def calculate_levels(group):
            """
            Calculates trading volume and price levels, specifically the Point of Control (POC), High Volume Node (HVN),
            and Low Volume Node (LVN) for a given group of market data. This function is designed to be applied to each 
            group of data within the DataFrame, typically grouped by date or another relevant identifier.

            The POC is identified as the price level with the highest trading volume, HVN as the second highest, and LVN 
            as the lowest volume level within the group. These levels are critical for understanding market behavior and 
            liquidity distribution.

            Args:
                group (pd.DataFrame): A subset of the original DataFrame representing a group of market data, expected to 
                contain 'TotalVolume' and 'Price' columns.

            Returns:
                pd.Series: A series containing the calculated POC, HVN, and LVN price and volume levels for the group.
            """
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





        combined_df = grouped.apply(calculate_levels) # Apply function to each group and combine results
        self.df = self.df.merge(combined_df, left_index=True, right_index=True, how='left') # Merge with Original DataFrame
        self.df.ffill(inplace=True) # Handle Missing Values
        #print(self.df.tail(50))
        print(f"After processing footprint dataframe, DataFrame shape: {self.df.shape}")
        
        return self.df.dropna()
    
