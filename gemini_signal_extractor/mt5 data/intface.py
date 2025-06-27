from typing import Optional
import MetaTrader5 as mt5
import json
import logging
import pandas as pd
import os

def setup_logging(log_path: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levellevellevel)s - %(message)s', filename=log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levellevellevel)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class MT5Interface:
    def __init__(self, config):
        self.config = config
        self.symbols = self.config['symbols'] if isinstance(self.config['symbols'], list) else [self.config['symbols']]
        self.lot = self.calculate_lot_size()
        self.connected = False

    def connect(self):
        uname = int(self.config['username'])
        pword = str(self.config['password'])
        trading_server = str(self.config['server'])
        filepath = str(self.config.get('path', ''))

        try:
            if mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
                logging.info("Trading Bot Starting")
                if mt5.login(login=uname, password=pword, server=trading_server):
                    logging.info("Trading Bot Logged in and Ready to Go!")
                    self.connected = True
                    return True
                else:
                    logging.error("MT5 Login Failed")
                    return False
        except Exception as e:
            logging.error(f"Error initializing MT5: {e}")
            return False

    def reconnect(self):
        logging.info("Attempting to reconnect to MT5...")
        if not self.connect():
            logging.error("Reconnection attempt failed")
            return False
        logging.info("Reconnected to MT5 successfully")
        return True

    def calculate_lot_size(self):
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Unable to fetch account information.")
            return 0.1
        
        risk_percentage = 0.01
        balance = account_info.balance
        lot_size = (balance * risk_percentage) / 1000
        logging.info(f"Calculated lot size: {lot_size}")
        return max(0.01, lot_size)

    def get_latest_data(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        if not self.connected:
            if not self.reconnect():
                return pd.DataFrame()

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logging.error(f"No historical data found for {symbol}.")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df

        except Exception as e:
            logging.error(f"Unexpected error getting latest data for {symbol}: {e}")
            return pd.DataFrame()

    def shutdown(self):
        logging.info("Shutting down MT5 connection")
        mt5.shutdown()
        self.connected = False
