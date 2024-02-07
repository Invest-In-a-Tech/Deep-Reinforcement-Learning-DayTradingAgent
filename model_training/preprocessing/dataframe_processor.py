##############################################
# IMPORTS 
# ============================================
import pandas as pd




############################################
# DataFrameProcessor
# ==========================================
class DataFrameProcessor:
    """
    A utility class for loading and preprocessing data from a CSV file. It is designed to read financial or time-series data, 
    convert date strings to datetime objects, set the date column as the DataFrame's index, and remove rows with missing values.

    Attributes:
        file_path (str): The path to the CSV file containing the data to be processed.
        df (pd.DataFrame): The pandas DataFrame holding the processed data. Initially set to None until the process_data method is called.

    Methods:
        process_data: Reads the CSV file specified by file_path, converts the 'Date' column to datetime, sets it as the index, 
        drops rows with missing values, and updates the df attribute with the processed DataFrame.
    """
    def __init__(self, file_path):
        """
        Initializes the DataFrameProcessor with the path to the data file.

        Parameters:
            file_path (str): The path to the CSV file to be processed.
        """
        self.file_path = file_path
        self.df = None




    ########################################
    # Process Data
    # ======================================
    def process_data(self):
        """
        Processes the CSV file by reading it into a pandas DataFrame, converting the 'Date' column to datetime, setting this column as the index,
        and removing any rows with missing values. This method prepares the data for analysis or modeling by ensuring it is in a clean and 
        structured format.

        Returns:
            pd.DataFrame: The processed DataFrame with 'Date' as the index and no missing values.
        """
        self.df = pd.read_csv(self.file_path) # Read CSV file into DataFrame
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d %H:%M:%S') # Convert 'Date' column to datetime
        self.df = self.df.set_index('Date') # Set 'Date' as the index
        self.df = self.df.dropna() # Drop rows with missing values
        print(f"After processing data, DataFrame shape: {self.df.shape}") # Print the shape of the processed DataFrame
        return self.df # Return the processed DataFrame
  


            
# Example usage            
#processor = DataFrameProcessor(file_path=os.path.join("model_training", "data", "es_tuples_dataset.csv"))
#df = processor.process_data()
#print(f"Heads of DataFrame: {self.df.head()}")
#print(f"Tails of DataFrame: {self.df.tail()}")