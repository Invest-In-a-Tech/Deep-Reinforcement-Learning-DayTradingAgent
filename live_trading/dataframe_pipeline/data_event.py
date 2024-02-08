# In this file, we are subscribing to tuples data and processing it into a DataFrame
############################################################
# IMPORTS 
# ==========================================================
import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv




############################################################
# Load environment variables
# ==========================================================
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")




############################################################
# DataEvent class
# ==========================================================
class DataEvent:
    """
    Manages the subscription and processing of streaming data from Sierra Charts. This class is designed to 
    interface with Sierra Charts, requesting streaming data based on specified parameters and processing the 
    incoming data for analysis or algorithmic trading.

    The class facilitates the initiation of data streams with detailed configurations such as the inclusion of 
    Volume by Price (VBP) data, the number of initial historical bars, and the frequency of real-time updates. 
    It ensures the data is structured and cleaned, making it ready for financial analysis tasks or as input for 
    trading algorithms.

    Attributes:
        sc: The Sierra Chart connection or session context used to request and receive data streams.
    """
    def __init__(self, sc):
        """
        Initializes the DataEvent class with a Sierra Chart session context and subscribes to data streams.

        Parameters:
            sc: The Sierra Chart connection or session context through which data requests are made.
        """
        self.sc = sc
        self.subscribe()




    ############################################################
    # subscribe
    # ==========================================================
    def subscribe(self):
        """
        Configures and initiates a data stream subscription from Sierra Charts. Specifies parameters such as 
        the number of bars for historical and real-time data, and whether to include Volume by Price (VBP) 
        data in the stream.
        """
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




    ############################################################
    # process_data_event
    # ==========================================================
    def process_dataframe(self, df):
        """
        Cleans and prepares a DataFrame received from Sierra Charts. Sorts data by date and price, 
        converts dates to datetime objects, sets dates as the index, and drops rows with missing values.

        Parameters:
            df (pd.DataFrame): The DataFrame containing Sierra Charts data to be processed.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame, ready for analysis or trading algorithms.
        """
        df = df.sort_values(by=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('Date', inplace=True)
        df = df.dropna()  # Remove rows with missing values
        return df




    ############################################################
    # process_data_event
    # ==========================================================
    def process_data_event(self, msg):
        """
        Processes a single data event from Sierra Charts. Extracts and transforms the data within the event message,
        including exploding tuple-based data into individual columns, and prepares it for analysis.

        Parameters:
            msg: The data event message containing raw data from Sierra Charts.

        Returns:
            pd.DataFrame: A DataFrame of processed data, structured for further analysis or algorithmic processing.
        """
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




        ###########################################
        # Explode Tuples
        # =========================================
        exploded_df = msg.df.explode('Tuples') # Explode Tuples into separate rows
        tuple_cols = ['Price', 'Bid', 'Ask', 'TotalVolume', 'NumberOfTrades'] # Columns within Tuples
        exploded_df[tuple_cols] = pd.DataFrame(exploded_df['Tuples'].tolist(), index=exploded_df.index) # Split Tuples into separate columns
        exploded_df.drop(columns=['BarNumber', 'Tuples'], inplace=True) # Remove unnecessary columns
        print(f"After processing data, DataFrame shape: {exploded_df.shape}")
        return self.process_dataframe(exploded_df) # Process the DataFrame

