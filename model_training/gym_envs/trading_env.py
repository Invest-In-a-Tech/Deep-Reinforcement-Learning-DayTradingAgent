import gym
from gym import spaces
import numpy as np
from datetime import datetime


class TradingEnv(gym.Env):
    def __init__(
        self, 
        features, 
        initial_balance=50000, 
        tick_value=12.50, 
        stop_loss=1000, 
        profit_target=1000, 
        start_time="08:30:00", 
        end_time="14:30:00",
    ):
        
        super(TradingEnv, self).__init__()
        
        # Data
        self.features = features.values
        #print(f'Features: {self.features}')
        self.feature_columns = features.columns
        self.dates = features.index
        
        # Define trading hours
        self.start_time = datetime.strptime(start_time, "%H:%M:%S").time()
        self.end_time = datetime.strptime(end_time, "%H:%M:%S").time()
        
        # Validate start and end times
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be earlier than end time.")


        # Initialize parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.tick_value = tick_value
        self.high_water_mark = initial_balance
        self.stop_loss = stop_loss
        self.profit_target = profit_target

        # Initialize state variables
        self.position = None  # 'long', 'short', or None
        self.entry_balance = None
        self.entry_price = None
        self.last_trade_reward = 0
        self.stop_loss_flag = 0  # Initialized stop loss flag

        # Sort DataFrame by index (Date) and extract unique timestamps
        self.features = features.sort_index()
        #print(f'Features: {self.features}')
        self.unique_timestamps = self.features.index.unique()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # Buy Entry, Buy Exit, Short Entry, Short Exit, No Action

        # Adjust observation space to include market data and account information
        market_data_shape = features.shape[1]
        additional_info_shape = 6  # balance, position flags, stop loss flag, open PnL, drawdown
        total_observation_shape = market_data_shape + additional_info_shape
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(total_observation_shape,), dtype=np.float32)
        

        # Reset environment to start state
        self.reset()


    def reset(self):
        self.current_timestamp_index = 0
        self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]
        self.current_row = 0
        self.balance = self.initial_balance  # Reset balance
        self.position = None
        self.entry_price = None
        self.high_water_mark = self.initial_balance
        self.entry_step = None
        self.stop_loss_flag = 0  # Reset stop loss flag
        return self._next_observation()
    
    
    def _next_observation(self):
        # Extract market data for the current timestamp
        filtered_df = self.features.loc[[self.current_timestamp]]

        # Check if current_row is within the bounds of filtered_df
        if self.current_row < len(filtered_df):
            row = filtered_df.iloc[self.current_row]
            market_data = row.values

            
            self.current_market_price = row['Price']

            # Calculate Open PnL
            if self.position == 'long':
                self.open_pnl = (self.current_market_price - self.entry_price) * self.tick_value
            elif self.position == 'short':
                self.open_pnl = (self.entry_price - self.current_market_price) * self.tick_value
            else:
                self.open_pnl = 0

            # Calculate drawdown from open_pnl
            if self.position is not None:
                self.drawdown = min(self.open_pnl, 0)
            else:
                self.drawdown = 0

            # Update stop loss flag
            if self.balance <= (self.high_water_mark - self.stop_loss):
                self.stop_loss_flag = 1
            else:
                self.stop_loss_flag = 0

            # Combine market data with account information
            position_flags = np.array([int(self.position == 'long'), int(self.position == 'short')])
            additional_info = np.array([self.balance, *position_flags, self.stop_loss_flag, self.open_pnl, self.drawdown])
            observation = np.concatenate((market_data, additional_info))
            return observation
        else:
            # Handle the case when current_row exceeds the length of filtered_df
            # Move to the next timestamp or handle the end of dataset
            self.current_timestamp_index += 1
            if self.current_timestamp_index >= len(self.unique_timestamps):
                # Handle the end of the dataset (e.g., by resetting the environment)
                return self.reset()
            else:
                # Move to the next timestamp and reset current_row
                self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]
                self.current_row = 0
                return self._next_observation()  


    def step(self, action):
        reward = 0
        done = False
        info = {}
        
        # Define transaction cost
        transaction_cost = 10  # Example fixed cost per trade        
        
        # Extract the current time
        current_time = self.current_timestamp.time()
     
        # End of Day Exit (outside of trading hours)
        if current_time > self.end_time:
            if self.position == 'long' or self.position == 'short':
                if self.position == 'long':
                    action = 1  # Force Buy Exit
                elif self.position == 'short':
                    action = 3  # Force Short Exit
                self.position = None
                self.entry_price = None
                self.entry_balance = None
                self.drawdown = 0
                #self.open_pnl = 0
                #print("End of Day Exiting all positions")
                
        # Check if the current time is within the trading hours
        if self.start_time <= current_time <= self.end_time:                

            # Check if the current time is within the trading hours
            if self.position is None:
                    # Buy Entry
                    if action == 0:
                        self.position = 'long'
                        self.entry_price = self.current_market_price
                        reward -= transaction_cost  # Applying transaction cost

                    # Short Entry
                    elif action == 2:
                        self.position = 'short'
                        self.entry_price = self.current_market_price
                        reward -= transaction_cost  # Applying transaction cost

                    # Do Nothing
                    elif action == 4:
                        pass

            elif self.position == 'long':
                # Buy Exit
                if action == 1:
                    pnl = (self.current_market_price - self.entry_price) * self.tick_value
                    self.balance += pnl
                    reward += pnl
                    self.position = None
                    reward -= transaction_cost  # Applying transaction cost

            elif self.position == 'short':
                # Short Exit
                if action == 3:
                    pnl = (self.entry_price - self.current_market_price) * self.tick_value
                    self.balance += pnl
                    reward += pnl
                    self.position = None
                    reward -= transaction_cost  # Applying transaction cost

        # Risk management penalty
        # Example risk penalty (can be customized)
        
        risk_penalty = 0
        
        if self.position is not None and (self.current_market_price - self.entry_price) * self.tick_value < -500:
            risk_penalty = 100
        
        reward -= risk_penalty

        
        # Sequential event processing
        # Increment the row index within the current timestamp
        self.current_row += 1

        # Check if all rows for the current timestamp are processed
        current_timestamp_rows = len(self.features.loc[[self.current_timestamp]])
        if self.current_row >= current_timestamp_rows or self.current_timestamp_index >= len(self.unique_timestamps) - 1:
            # Move to the next timestamp
            self.current_timestamp_index += 1
            if self.current_timestamp_index >= len(self.unique_timestamps):
                done = True
            else:
                self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]
                self.current_row = 0
        else:
            # Update current market price if still within the current timestamp
            self.current_market_price = self.features.loc[self.current_timestamp].iloc[self.current_row]['Price']

        observation = self._next_observation()

        return observation, reward, done, info


    def render(self, mode='human', action=None, reward=None):
        if mode == 'human':
            # Human-readable printout of the current state in a single line
            state_info = (
                f"Timestamp: {self.current_timestamp}, "
                f"Current Row: {self.current_row}, "
                f"Current Position: {self.position}, "
                f"Action: {action},"
                f"Balance: {self.balance}, "
                f"Entry Price: {self.entry_price}, "
                f"Current Market Price: {self.current_market_price}, "
                f"Open PnL: {self.open_pnl},"
                f"Reward: {reward}, "
                f"Drawdown: {self.drawdown}, "
            )
            print(state_info)
        else:
            raise NotImplementedError("Only 'human' mode is supported for rendering.")
        
