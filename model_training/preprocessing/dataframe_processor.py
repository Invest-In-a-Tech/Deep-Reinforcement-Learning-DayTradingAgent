import pandas as pd
import os
class DataFrameProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None


    def process_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d %H:%M:%S')
        self.df = self.df.set_index('Date')
        self.df = self.df.dropna() 
        #print(f"Heads of DataFrame: {self.df.head()}")
        #print(f"Tails of DataFrame: {self.df.tail()}")
        print(f"After processing data, DataFrame shape: {self.df.shape}")
        return self.df
  
            
# Example usage            
#processor = DataFrameProcessor(file_path=os.path.join("model_training", "data", "es_tuples_dataset.csv"))
#df = processor.process_data()