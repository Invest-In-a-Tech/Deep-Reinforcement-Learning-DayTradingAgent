import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())
sc_api_key = os.environ.get("SC_API_KEY")


class PositionEvent:
    def __init__(self, sc):
        self.sc = sc
        self.pos_reqid = None
        self.subscribe()

        self.position = None
        self.avg_price = None
        self.open_pnl = 0
        
        self.cumulative_pnl = 0.0
        self.previous_open_pnl = 0.0       
       
        
    def subscribe(self):
        self.pos_reqid = self.sc.get_position_status(key=sc_api_key, subscribe=True)


    def process_position_event(self, msg):
        #trade position
        position = 1.0 if msg.qty > 0 else -1.0 if msg.qty < 0 else 0.0
        
        # If a position is still open
        if position != 0:
            change_in_pnl = msg.open_pnl - self.previous_open_pnl
            self.cumulative_pnl += change_in_pnl
            self.previous_open_pnl = msg.open_pnl
        # If position is closed
        else:
            previous_open_pnl = 0.0
        
        data = {
            'position': position,
            'avg_price': msg.avg_price,
            'open_pnl': msg.open_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'previous_open_pnl': self.previous_open_pnl,  
        }
        self.position = data['position']
        self.avg_price = data['avg_price']
        self.open_pnl = data['open_pnl']   
        self.cumulative_pnl = data['cumulative_pnl']
        self.previous_open_pnl = data['previous_open_pnl']  
        return data
