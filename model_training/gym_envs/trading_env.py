##############################################
################# IMPORTS ####################
import gym
from gym import spaces
import numpy as np
from datetime import datetime


#######################################################
###### TRADING ENVIRONMENT FOR FINANCIAL MARKETS ######
class TradingEnv(gym.Env):
    """
    A custom trading environment class compatible with OpenAI's gym interface.
    It simulates a trading scenario allowing actions like buying, selling, and holding.

    Parameters:
    - features (DataFrame): Historical market data features.
    - initial_balance (float): The starting balance for the trading account.
    - tick_value (float): The value of each market tick.
    - stop_loss (float): The stop loss amount.
    - profit_target (float): The profit target.
    - start_time (str): The opening time of the trading session.
    - end_time (str): The closing time of the trading session.
    """
    
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

        # Process and set the market data features
        self.features = features.values
        self.feature_columns = features.columns
        self.dates = features.index
        
        # Setting up trading session timings
        self.start_time = datetime.strptime(start_time, "%H:%M:%S").time()
        self.end_time = datetime.strptime(end_time, "%H:%M:%S").time()
        
        # Validate the trading session timings
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be earlier than end time.")

        # Initialize variables for account management
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.tick_value = tick_value
        self.high_water_mark = initial_balance
        self.stop_loss = stop_loss
        self.profit_target = profit_target

        # Position management variables
        self.position = None 
        self.entry_balance = None
        self.entry_price = None
        self.last_trade_reward = 0
        self.stop_loss_flag = 0 

        # Organize market data
        self.features = features.sort_index()
        self.unique_timestamps = self.features.index.unique()

        # Define the action space (5 actions: buy entry/exit, short entry/exit, no action)
        self.action_space = spaces.Discrete(5) 

        # Define the observation space (market data + account information)
        market_data_shape = features.shape[1]
        additional_info_shape = 6 # Balance, position flags, stop loss flag, open PnL, drawdown
        total_observation_shape = market_data_shape + additional_info_shape
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(total_observation_shape,), dtype=np.float32)
        
        # Initialize the environment to its start state
        self.reset()


    #####################################################
    ####### RESET METHOD FOR TRADING ENVIRONMENT ########
    def reset(self):
        """
        Resets the trading environment to its initial state.
        This method is typically called at the beginning of each new episode.

        Returns:
        - The initial observation of the environment.
        """
        
        # Reset the index for the current timestamp to the start of the dataset
        self.current_timestamp_index = 0
        self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]

        # Reset the row counter for the current state in the dataset
        self.current_row = 0

        # Reinitialize the account balance to the initial balance
        self.balance = self.initial_balance

        # Reset position-related variables to None, indicating no current position
        self.position = None
        self.entry_price = None
        self.high_water_mark = self.initial_balance  # Reset the high water mark to the initial balance

        # Reset the step at which the current position was entered (if any)
        self.entry_step = None

        # Reset the stop loss flag, used to indicate if stop loss was triggered in the last step
        self.stop_loss_flag = 0

        # Generate the next observation after the reset
        return self._next_observation()
    
    
    #######################################################
    ############# NEXT OBSERVATION METHOD #################
    def _next_observation(self):
        """
        Generates the next observation of the market and the agent's state.

        This method is called to update the state of the environment after each action.

        Returns:
        - The next observation, which includes both market data and account information.
        """

        # Extract the market data for the current timestamp from the features DataFrame
        filtered_df = self.features.loc[[self.current_timestamp]]

        # Ensure that the current row is within the bounds of the filtered data
        if self.current_row < len(filtered_df):
            row = filtered_df.iloc[self.current_row]
            market_data = row.values

            # Update the current market price from the 'Price' column
            self.current_market_price = row['Price']

            # Calculate the open profit and loss (PnL) based on the current position
            if self.position == 'long':
                self.open_pnl = (self.current_market_price - self.entry_price) * self.tick_value
            elif self.position == 'short':
                self.open_pnl = (self.entry_price - self.current_market_price) * self.tick_value
            else:
                self.open_pnl = 0

            # Calculate the drawdown from the open PnL
            self.drawdown = min(self.open_pnl, 0) if self.position is not None else 0

            # Update the stop loss flag based on account balance and high water mark
            self.stop_loss_flag = 1 if self.balance <= (self.high_water_mark - self.stop_loss) else 0

            # Combine market data with additional account information for the observation
            position_flags = np.array([int(self.position == 'long'), int(self.position == 'short')])
            additional_info = np.array([self.balance, *position_flags, self.stop_loss_flag, self.open_pnl, self.drawdown])
            observation = np.concatenate((market_data, additional_info))
            return observation
        else:
            # Handle the case when the current row exceeds the length of filtered data
            self.current_timestamp_index += 1
            if self.current_timestamp_index >= len(self.unique_timestamps):
                # Reset the environment if the end of the dataset is reached
                return self.reset()
            else:
                # Move to the next timestamp and reset the current row
                self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]
                self.current_row = 0
                return self._next_observation()


    ##############################################
    ############# STEP METHOD ####################
    
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
                        #reward -= transaction_cost  # Applying transaction cost

                    # Short Entry
                    elif action == 2:
                        self.position = 'short'
                        self.entry_price = self.current_market_price
                        #reward -= transaction_cost  # Applying transaction cost

                    # Do Nothing
                    elif action == 4:
                        pass

            elif self.position == 'long':
                # Buy Exit
                if action == 1:
                    pnl = (self.current_market_price - self.entry_price) * self.tick_value
                    self.balance += pnl
                    reward += pnl * 10
                    self.position = None
                    #reward -= transaction_cost  # Applying transaction cost

            elif self.position == 'short':
                # Short Exit
                if action == 3:
                    pnl = (self.entry_price - self.current_market_price) * self.tick_value
                    self.balance += pnl
                    reward += pnl * 10
                    self.position = None
                    #reward -= transaction_cost  # Applying transaction cost

        # Risk management penalty
        # Example risk penalty (can be customized)
        
        risk_penalty = 0
        
        if self.position is not None and (self.current_market_price - self.entry_price) * self.tick_value < -500:
            risk_penalty = 10
        
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


    ##############################################
    ############# RENDER METHOD ##################
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
        
