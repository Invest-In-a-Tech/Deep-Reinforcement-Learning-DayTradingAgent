##############################################
# IMPORTS 
# ============================================
import gym
from gym import spaces
import numpy as np
from datetime import datetime




#######################################################
# TRADING ENVIRONMENT FOR FINANCIAL MARKETS SIMULATION
# =====================================================
class TradingEnv(gym.Env):
    """
    A simulation environment for trading ES emini futures contracts using Footprint data, built on the OpenAI Gym framework.
    This environment facilitates the development, testing, and evaluation of algorithmic trading strategies by providing a
    realistic trading session experience with discrete actions for entering and exiting long and short positions, as well as the
    option to take no action.

    The environment processes historical Footprint data, which includes multiple rows per timestamp, to simulate the dynamics of
    the futures market. Traders can implement strategies that respond to fine-grained market changes within each trading session,
    defined by start and end times.

    Parameters:
        features (DataFrame): A pandas DataFrame containing the Footprint data for simulation, indexed by timestamps.
        initial_balance (float): The starting balance of the trading account.
        tick_value (float): The profit or loss per tick movement in the price of the futures contract.
        stop_loss (float): The loss threshold that triggers a forced exit from a position.
        profit_target (float): The profit goal at which a position is closed to realize gains.
        start_time (str): The start time of the simulated trading session.
        end_time (str): The end time of the simulated trading session.

    The environment defines an action space of 5 discrete actions and an observation space that includes both market data
    from the Footprint data and account information. It supports the evaluation of trading strategies over the course of
    multiple simulated trading days, enabling traders to refine their approaches based on historical data.

    This specialized environment is tailored for those interested in exploring the intricacies of trading ES emini futures
    contracts, providing a platform for rigorous strategy testing without the financial risk of live market trading.
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
    #   RESET METHOD 
    # ===================================================
    def reset(self):
        """
        Resets the trading environment to its initial state. This involves reinitializing the time index,
        account balance, trading positions, and other relevant variables to their starting values. The method
        is typically called at the beginning of a new episode or simulation run to ensure a clean start.

        The reset process includes:
        - Setting the current timestamp index back to the beginning of the dataset.
        - Resetting the current row counter to ensure the simulation starts from the first row of data.
        - Reinitializing the account balance to the initial balance specified at the creation of the environment.
        - Clearing any existing trading positions, entry prices, and setting the high water mark to the initial balance.
        - Resetting the stop loss flag and any other flags or counters used in the simulation.

        Returns:
            The initial observation from the environment after the reset, allowing an agent to start
            interacting with the environment immediately with a clear understanding of the initial state.

        This method ensures that the environment is in a consistent, known state at the beginning of each episode,
        facilitating reproducible experiments and allowing for the evaluation of trading strategies over multiple iterations.
        """
        self.current_timestamp_index = 0  # Reset the index for the current timestamp to the start of the dataset
        self.current_timestamp = self.unique_timestamps[self.current_timestamp_index]
        self.current_row = 0  # Reset the row counter for the current state in the dataset
        self.balance = self.initial_balance # Reinitialize the account balance to the initial balance
        self.position = None # Reset position-related variables to None, indicating no current position
        self.entry_price = None
        self.high_water_mark = self.initial_balance  # Reset the high water mark to the initial balance
        self.entry_step = None # Reset the step at which the current position was entered (if any)
        self.stop_loss_flag = 0 # Reset the stop loss flag, used to indicate if stop loss was triggered in the last step

        # Generate the next observation after the reset
        return self._next_observation()
    
    
    
    
    #######################################################
    # NEXT OBSERVATION METHOD 
    # =====================================================
    def _next_observation(self):
        """
        Generates the next observation from the environment by extracting market data for the current timestamp,
        calculating open profit and loss (PnL), drawdown, and updating account-related metrics such as stop loss flag.
        This method prepares a comprehensive state representation that includes both market data and account status
        for the reinforcement learning agent to use in making decisions.

        The observation is a combination of raw market data from the current time step and derived metrics that
        include the current balance, position status (long/short), stop loss flag, open PnL, and drawdown. The method
        ensures the observation is within the bounds of the available data and handles the transition between timestamps
        or resets the environment when necessary.

        Returns:
            np.ndarray: The next observation as a numpy array, combining raw market data with calculated metrics
            and account status. If the end of the dataset is reached, it may trigger a reset of the environment.

        The method performs several key operations:
        - Extracts the market data for the current timestamp.
        - Updates the current market price and calculates the open PnL based on the current position (long/short).
        - Calculates the drawdown from the open PnL.
        - Updates the stop loss flag based on the current balance and a predefined high water mark.
        - Combines the market data with additional account information to form the observation.
        - Handles the case where the current row exceeds the available data for the current timestamp by moving
        to the next timestamp or resetting the environment if the dataset's end is reached.
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
    # STEP METHOD 
    # ============================================
    def step(self, action):
        """
        Executes one step in the trading environment based on the given action, updates the environment's state,
        and calculates the reward resulting from the action. It handles trading actions such as entering and exiting
        positions, enforces end-of-day position liquidation outside trading hours, and applies transaction costs.

        Args:
            action: An integer representing the action to be taken. Actions could include entering a long position (0),
            exiting a long position (1), entering a short position (2), exiting a short position (3), or doing nothing (4).

        Returns:
            A tuple containing:
            - The immediate reward as a result of the action taken.
            - A boolean indicating whether the episode (trading day) has ended (`done`).
            - A dictionary (`info`) with additional information for debugging or detailed analysis.

        The method performs several key functions:
        - It applies a fixed transaction cost to each trade.
        - It checks if the current time is outside trading hours to force exit any open positions.
        - Within trading hours, it processes the specified action, updating positions and calculating open profit or loss.
        - It calculates rewards based on the action's outcome, including penalizing missed opportunities or rewarding correct market anticipations.
        - Updates the agent's balance with the reward from the trade.
        """
        reward = 0
        done = False
        info = {}
        trade_reward = 0
        
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
                trade_reward = self.open_pnl
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
                        self.open_pnl = 0

                    # Short Entry
                    elif action == 2:
                        self.position = 'short'
                        self.entry_price = self.current_market_price
                        self.open_pnl = 0

                    # Do Nothing
                    elif action == 4:
                        # Check if the market is flat
                        if abs(self.current_market_price - self.current_market_price - 1) <= 2:  # Assuming a small threshold for flat market
                            reward += 50
                        # Check for missed opportunity
                        else:
                            reward -= 50
                            
            elif self.position == 'long':
                # Buy Exit
                if action == 1:
                    trade_reward = (self.current_market_price - self.entry_price) * self.tick_value
                    self.position = None
                    self.entry_price = None
                    self.drawdown = 0
                    self.open_pnl = 0

            elif self.position == 'short':
                # Short Exit
                if action == 3:
                    trade_reward = (self.entry_price - self.current_market_price) * self.tick_value
                    self.position = None
                    self.entry_price = None
                    self.drawdown = 0
                    self.open_pnl = 0                   
                    



        ##################################################
        # REWARD CALCULATION
        # ================================================
        """
        """
        reward += trade_reward
        self.balance += reward       

            
            
            
        ##################################################
        # SEQUENTIAL EVENT PROCESSING 
        # ================================================
        """
        This method processes events in a sequential manner based on timestamps. 
        It is designed to iterate over rows (events) associated with the current timestamp,
        updating the state of the system with each event. Once all events for a given timestamp
        are processed, it moves to the next timestamp. The method ensures that events are processed
        in temporal order and updates the system's state accordingly.

        The processing involves incrementing the row index to move through events within the same timestamp.
        If all events for the current timestamp have been processed, or if we are at the last timestamp in the dataset,
        the method progresses to the next timestamp. It also updates the current market price based on the latest event
        within the current timestamp, ensuring the system's state reflects the most recent data.

        Returns:
            tuple: A tuple containing the next observation, the reward associated with the transition,
            a boolean indicating whether the episode is done, and a dictionary with additional information.

        The method assumes the presence of instance variables such as `current_row`, `current_timestamp`, and
        `current_market_price` to maintain the state of the simulation or processing. It relies on `features`,
        a DataFrame or similar structure, to access event data indexed by timestamps.
        """
        # Increment the row index within the current timestamp
        self.current_row += 1
        
        # Check if all rows for the current timestamp are processed
        current_timestamp_rows = len(self.features.loc[[self.current_timestamp]])
        if self.current_row >= current_timestamp_rows or self.current_timestamp_index >= len(self.unique_timestamps) - 1:
            self.current_timestamp_index += 1  # Move to the next timestamp
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
    # RENDER METHOD 
    # ============================================
    def render(self, mode='human', action=None, reward=None):
        """
        Provides a visualization or a human-readable output of the current state of the system.
        This method is primarily used for debugging and monitoring purposes, allowing users
        to view the current state, including the timestamp, position, action taken, account balance,
        entry price, current market price, open PnL (Profit or Loss), reward received, and any drawdown.

        Args:
            mode (str, optional): The rendering mode. Currently, only 'human' mode is implemented, which
                prints the state information in a human-readable format. Defaults to 'human'.
            action (optional): The last action taken by the agent. The type depends on the specific implementation
                of the trading system or environment. Defaults to None.
            reward (optional): The reward obtained from taking the previous action. This can be any numerical value
                reflecting the performance of the action according to the system's reward function. Defaults to None.

        Raises:
            NotImplementedError: If the rendering mode specified is not supported, this exception is raised.
                Currently, only 'human' mode is implemented, which provides a straightforward textual representation
                of the system's state.

        The method displays key information about the system's current state, including but not limited to:
        - The current timestamp indicating the point in time of the simulation or real-world data being processed.
        - The current row, which helps in understanding the progression through the dataset.
        - The agent's current position (e.g., long, short, or neutral) in the market.
        - The last action taken by the agent and the immediate reward received for that action.
        - The current balance, entry price, and market price, providing insights into the financial status.
        - Open PnL, showing the current profit or loss from open positions.
        - Drawdown, indicating the decline from a peak to a trough in the value of an account or an investment.
        """
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
        
