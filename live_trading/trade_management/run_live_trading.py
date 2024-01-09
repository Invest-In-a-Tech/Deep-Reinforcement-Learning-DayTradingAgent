from live_trading.trade_management.trade_data_processor import TradeDataProcessor

def run_live_trading():
    processor = TradeDataProcessor()
    processor.process_data()