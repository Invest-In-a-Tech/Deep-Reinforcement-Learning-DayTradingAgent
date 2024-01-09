import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


class AccountEvent:
    def __init__(self, sc):
        self.sc = sc
        self.acct_reqid = None
        self.subscribe()
        
        self.available_funds = None 
        self.trade_account = None
        self.open_positions_pnl = None
        self.daily_pnl = None
        self.cash_balance = None
         
               
    def subscribe(self):
        self.acct_reqid = self.sc.get_account_status(key=sc_api_key, subscribe=True)
        #print(self.msg.dict) 


    def process_account_event(self, msg):
        #print(msg) 
        data = {
            'available_funds': msg.available_funds,
            'trade_account': msg.trade_account,
            'cash_balance': msg.cash_balance,
            'open_positions_pnl': msg.open_positions_pnl,
            'daily_pnl': msg.daily_pnl
        }
        self.available_funds = data['available_funds']
        self.trade_account = data['trade_account']
        self.cash_balance = data['cash_balance']
        self.open_positions_pnl = data['open_positions_pnl']
        self.daily_pnl = data['daily_pnl']        
        return data
