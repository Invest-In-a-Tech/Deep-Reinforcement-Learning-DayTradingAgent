import joblib
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from trade29.sc.bridge import SCBridge

from live_trading.trade_management.model_handling import ModelHandler
from live_trading.trade_management.account_manager import AccountManager
from live_trading.trade_management.trading_logic import execute_trading_logic

from live_trading.events.account_event import AccountEvent
from live_trading.events.position_event import PositionEvent

from live_trading.dataframe_pipeline.data_event import DataEvent
from live_trading.dataframe_pipeline.footprint_dataframe import FootprintDataframe
from live_trading.dataframe_pipeline.main_dataframe import MainDataframe

import os
from dotenv import find_dotenv, load_dotenv


# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


class TradeDataProcessor:
    def __init__(self, start_time="08:30:00", end_time="14:30:00"):
        
        # Create an instance of SCBridge
        self.sc = SCBridge()

        # Pass the SCBridge instance to our DataEvent, AccountEvent, PositionEvent
        self.data_event = DataEvent(self.sc)
        self.account_event = AccountEvent(self.sc)
        self.position_event = PositionEvent(self.sc)
          
        self.start_time = datetime.strptime(start_time, "%H:%M:%S").time()
        self.end_time = datetime.strptime(end_time, "%H:%M:%S").time()
        
        # Initialize ModelHandler
        self.model_handler = ModelHandler(
            os.path.join("model_training", "training", "saved_models", "PPO_fold_1.zip"), 
            os.path.join("model_training", "training", "saved_scalers", "scaler_fold_1")
        )
                
        # Initialize account manager
        self.account_manager = AccountManager()     
        
        
    def prepare_data(self, data_message):
        # Extract and process the data from the message
        self.df = self.data_event.process_data_event(data_message)

        # Process the data frame for the model
        footprint_dataframe = FootprintDataframe(self.df)
        df_enhanced = footprint_dataframe.process_footprint_dataframe()
        return df_enhanced
    
    
    def manage_account(self):
        # Logic for managing account
        self.account_info = self.account_manager.manage_account(self.position_event)
        return self.account_info
        
                   
    def process_data(self):
        # Start an infinite loop       
        while True:
            # Get the data event
            self.msg = self.sc.get_response_queue().get()
            
            # Check if the request ID is for account data
            if self.msg.request_id == self.account_event.acct_reqid:
                self.account_event.process_account_event(self.msg)
           
            # Check if the request ID is for position data 
            elif self.msg.request_id == self.position_event.pos_reqid:
                self.position_event.process_position_event(self.msg)
           
            # Check if the request ID is for data
            elif self.msg.request_id == self.data_event.data_reqid:
                df_enhanced = self.prepare_data(self.msg) 
                print(df_enhanced)    
                
                # Call the manage_account method and get the updated account info
                self.account_info = self.manage_account()            
                  
                # Model handling
                obs = self.model_handler.prepare_observation(
                    df_enhanced, 
                    self.account_info['long_position'], 
                    self.account_info['short_position'], 
                    self.account_info['current_balance'], 
                    self.account_info['stop_loss_flag'], 
                    self.account_info['drawdown'], 
                    self.position_event.open_pnl,
                )
                action = self.model_handler.predict_action(obs)
                
                # Trading logic   
                current_time = self.df.index[-1].time()  
                
                # Call the execute_trading_logic function
                execute_trading_logic(
                    self.sc, 
                    self.account_info, 
                    self.df.index[-1].time(),  # current time
                    action, 
                    self.start_time, 
                    self.end_time
                )
                
                # Exit time - Check if the current time is past the exit time
                exit_time = datetime.strptime("14:31:00", "%H:%M:%S").time()  
                if current_time >= exit_time:
                
                    # Reset values for the next trading session
                    self.account_manager.reset_account(position_event=self.position_event)
    
                    # Check if there are any open positions
                    if self.account_info['current_position'] != 0:
                        self.account_info['current_position'] = 0
                        self.sc.flatten_and_cancel(key=sc_api_key)  # Cancel all open orders
                        print("End of Day: Exiting all positions")  

