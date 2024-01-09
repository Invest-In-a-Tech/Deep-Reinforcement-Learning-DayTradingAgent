# account_manager.py

class AccountManager:
    DEFAULT_BEGINNING_BALANCE = 50000
    DEFAULT_HIGH_WATER_MARK = 0.0
    STOP_LOSS_THRESHOLD = 1000
    PROFIT_TARGET_THRESHOLD = 1000


    def __init__(self):
        self.beginning_balance = self.DEFAULT_BEGINNING_BALANCE
        self.high_water_mark = self.DEFAULT_HIGH_WATER_MARK
        self.current_balance = None
        self.stop_trading = False
        self.stop_trading_after_profit_target = False
        self.current_position = None
        self.entry_balance = None


    def _initialize_balance(self, position_event):
        """Initializes the balance based on the position event."""
        if position_event.cumulative_pnl is None:
            print("Warning: cumulative_pnl is not initialized.")
            position_event.cumulative_pnl = 0.0
        print("cumulative_pnl:", position_event.cumulative_pnl)
        self.current_balance = self.beginning_balance + position_event.cumulative_pnl
        print(f"Account Value: {self.current_balance}")


    def _update_high_water_mark(self):
        """Updates the high water mark if the current balance exceeds it."""
        if self.current_balance > self.high_water_mark:
            self.high_water_mark = self.current_balance
        print(f"High water mark: {self.high_water_mark}")


    def _check_stop_conditions(self, position_event):
        """Checks and updates trading stop conditions."""
        stop_loss_flag = 0
        if self.current_balance <= (self.high_water_mark - self.STOP_LOSS_THRESHOLD):
            print("Agent Message: Stop loss reached. I'm no longer allowed to trade!")
            stop_loss_flag = 1
            self.stop_trading = True

        profit_target_flag = 0
        if position_event.cumulative_pnl >= self.PROFIT_TARGET_THRESHOLD:
            profit_target_flag = 1
            self.stop_trading_after_profit_target = True

        return stop_loss_flag, profit_target_flag


    def _update_position_status(self, position_event):
        """Updates the position status."""
        self.current_position = position_event.position
        long_position = int(self.current_position == 1.0)
        short_position = int(self.current_position == -1.0)
        print(f"Long position: {long_position}")
        print(f"Short position: {short_position}")

        if self.current_position in [1.0, -1.0]:
            self.entry_balance = self.current_balance
        else:
            self.entry_balance = None
        print(f"Entry balance: {self.entry_balance}")

        drawdown = min(position_event.open_pnl, 0) if position_event.position is not None else 0
        print(f"Drawdown: {drawdown}")
        print(f"Open PnL: {position_event.open_pnl}")

        return long_position, short_position, drawdown
    
    
    def reset_account(self, position_event):
        """Resets the account state for a new trading session."""
        #print(f"Resetting: Current Cumulative PnL: {position_event.cumulative_pnl}, High Water Mark: {self.high_water_mark}")
        position_event.cumulative_pnl = 0.0
        self.high_water_mark = self.DEFAULT_HIGH_WATER_MARK
        #print(f"Reset Complete: Cumulative PnL: {position_event.cumulative_pnl}, High Water Mark: {self.high_water_mark}")


    def manage_account(self, position_event):
        """Manages the account based on the given position event."""
        print(f"Beginning balance: {self.beginning_balance}")

        self._initialize_balance(position_event)
        self._update_high_water_mark()
        stop_loss_flag, profit_target_flag = self._check_stop_conditions(position_event)
        long_position, short_position, drawdown = self._update_position_status(position_event) 

        return {
            'stop_trading': self.stop_trading,
            'stop_trading_after_profit_target': self.stop_trading_after_profit_target,
            'current_position': self.current_position,
            'entry_balance': self.entry_balance,
            'drawdown': drawdown,
            'stop_loss_flag': stop_loss_flag,
            'profit_target_flag': profit_target_flag,
            'long_position': long_position,
            'short_position': short_position,
            'high_water_mark': self.high_water_mark,
            'cumulative_pnl': position_event.cumulative_pnl,
            'beginning_balance': self.beginning_balance,
            'stop_loss': self.STOP_LOSS_THRESHOLD,
            'current_balance': self.current_balance,
        }
