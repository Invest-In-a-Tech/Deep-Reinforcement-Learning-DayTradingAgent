# trading_logic.py

import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


def execute_trading_logic(sc, account_info, current_time, action, start_time, end_time):
    # Trading logic   
                
    if account_info['stop_trading'] and account_info['current_position'] != 0:
        sc.flatten_and_cancel(key=sc_api_key)  # Cancel all open orders
        print("Agent Message: Exiting due to stop trading condition being True")
        account_info["current_position"] = 0  
                    
    if account_info['stop_trading_after_profit_target'] and account_info['current_position'] != 0:
        sc.flatten_and_cancel(key=sc_api_key)
        print("Agent Message: Exiting due to stop trading after profit target condition being True")
        account_info['current_position'] = 0   
                    
    if start_time <= current_time <= end_time and not account_info['stop_trading'] \
       and not account_info['stop_trading_after_profit_target']:                                                                                                                                                                                                                                 
        if account_info['current_position'] == 0:
            
            # Buy entry
            if action == 0:
                print("Long condition")
                sc.submit_order(key=sc_api_key, qty=1, is_buy=True, target_enabled=True, target_offset=10, stop_enabled=True, stop_offset=10)
                account_info['current_position'] = 1
                print("Entry balance:", account_info["entry_balance"])
            
            # Short entry
            elif action == 2:
                print("Short condition")
                sc.submit_order(key=sc_api_key, qty=1, is_buy=False, target_enabled=True, target_offset=10, stop_enabled=True, stop_offset=10)
                account_info['current_position'] = -1
                print("Entry balance:", account_info['entry_balance'])
            
            # No action
            elif action == 4:
                print("No action")       
        
        # Long Position Open
        elif account_info['current_position'] == 1:
            
            # Buy exit
            if action == 1: 
                sc.flatten_and_cancel(key=sc_api_key)  # Cancel all open orders
                print("Long exit")
                account_info['current_position'] = 0
        
        # Short Position Open 
        elif account_info['current_position'] == -1: 
            
            # Short exit
            if action == 3:
                sc.flatten_and_cancel(key=sc_api_key)  # Cancel all open orders
                print("Short exit")
                account_info['current_position'] = 0
