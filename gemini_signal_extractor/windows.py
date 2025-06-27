import os
import pandas as pd
import yaml
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime, timedelta
import re

# List of valid Forex and crypto pairs
VALID_SYMBOLS = [
    # Major Forex Pairs
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'GBPCHF', 'AUDJPY', 'CADJPY',
    'NZDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'EURAUD', 'EURCAD',
    'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDCHF',
    
    # Minor Forex Pairs
    'EURHKD', 'EURMXN', 'EURNOK', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY',
    'EURZAR', 'GBPHKD', 'GBPMXN', 'GBPNOK', 'GBPPLN', 'GBPSEK', 'GBPSGD',
    'GBPTRY', 'GBPZAR', 'USDHKD', 'USDMXN', 'USDNOK', 'USDPLN', 'USDSEK',
    'USDSGD', 'USDTRY', 'USDZAR', 'AUDHKD', 'AUDMXN', 'AUDNOK', 'AUDPLN',
    'AUDSEK', 'AUDSGD', 'AUDTRY', 'AUDZAR', 'CADHKD', 'CADMXN', 'CADNOK',
    'CADPLN', 'CADSEK', 'CADSGD', 'CADTRY', 'CADZAR', 'CHFHKD', 'CHFMXN',
    'CHFNOK', 'CHFPLN', 'CHFSEK', 'CHFSGD', 'CHFTRY', 'CHFZAR', 'JPYHKD',
    'JPYMXN', 'JPYNOK', 'JPYPLN', 'JPYSEK', 'JPYSGD', 'JPYTRY', 'JPYZAR',
    'NZDHKD', 'NZDMXN', 'NZDPLN', 'NZDSEK', 'NZDSGD', 'NZDTRY', 'NZDZAR',
    
    # Commodities
    'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'XAUXAG', 'XAUXPT', 'XAUXPD',
    'XAGXPT', 'XAGXPD', 'XPTXPD', 'GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM',
    'BRENT', 'WTI', 'NATURALGAS', 'COPPER', 'ALUMINUM', 'NICKEL', 'ZINC',
    'LEAD', 'TIN', 'IRONORE', 'COAL', 'SUGAR', 'COFFEE', 'COTTON', 'CORN',
    'WHEAT', 'SOYBEAN', 'OIL',
    
    # Cryptocurrencies
    'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'EOSUSD', 'XLMUSD',
    'ADAUSD', 'DOTUSD', 'LINKUSD', 'UNIUSD', 'SOLUSD', 'AVAXUSD', 'MATICUSD',
    'DOGEUSD', 'SHIBUSD', 'BTCETH', 'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT',
    'BCHUSDT', 'EOSUSDT', 'XLMUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT',
    'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'DOGEUSDT', 'SHIBUSDT',
    
    # Indices
    'US30', 'US500', 'USTEC', 'UK100', 'GER40', 'FRA40', 'ESP35', 'ITA40',
    'JPN225', 'AUS200', 'HKG50', 'SIN50', 'SWI20', 'NLD25', 'SWE30', 'NOR25',
    'DJI', 'SPX', 'NDX', 'FTSE', 'DAX', 'CAC', 'IBEX', 'NIKKEI', 'ASX',
    'HSI', 'STI', 'SMI',
    
    # Common Variations
    'GOLDUSD', 'SILVERUSD', 'PLATINUMUSD', 'PALLADIUMUSD', 'OILUSD',
    'USOIL', 'UKOIL', 'BRENTUSD', 'WTIUSD', 'NATGASUSD', 'NATGAS',
    'COPPERUSD', 'COPPER', 'ALUMINUMUSD', 'ALUMINUM', 'NICKELUSD', 'NICKEL',
    'ZINCUSD', 'ZINC', 'LEADUSD', 'LEAD', 'TINUSD', 'TIN', 'IRONOREUSD',
    'IRONORE', 'COALUSD', 'COAL', 'SUGARUSD', 'SUGAR', 'COFFEEUSD', 'COFFEE',
    'COTTONUSD', 'COTTON', 'CORNUSD', 'CORN', 'WHEATUSD', 'WHEAT',
    'SOYBEANUSD', 'SOYBEAN'
]

# Signal detection keywords
SIGNAL_KEYWORDS = [
    'buy', 'sell', 'long', 'short',
    'buying', 'selling', 'bullish', 'bearish',
    'entry', 'enter', 'tp', 'sl', 
    'stop loss', 'take profit', 'target',
    'going up', 'going down', 'uptrend', 'downtrend'
]

class SignalExtractor:
    def __init__(self, config_path=None):
        """Initialize the signal extractor."""
        # Use absolute path for config.yaml
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        
        self.config = self.load_config(config_path)
        
        # Setup logging
        self.setup_logging()
        
        # Get data directory from config
        self.data_dir = r"C:\Users\PC\OneDrive\Desktop\python\gemini_signal_extractor\data"
        
        # Create necessary directories
        self.create_directories()
        
        # Get time window from config (in minutes)
        self.time_window_minutes = self.config['signal_extraction'].get('time_window_minutes', 2)
        
        # Filter out commented groups (those starting with #)
        self.groups = [group for group in self.config['telegram']['groups'] 
                      if not str(group).strip().startswith('#')]
        
        self.logger.info(f"Initialized with time_window_minutes={self.time_window_minutes}")
        self.logger.info(f"Active groups: {self.groups}")

    def load_config(self, config_path='config.yaml'):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            raise



    def setup_logging(self):
        """Setup logging with UTF-8 encoding."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'signal_extractor.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            'logs',
            os.path.join(self.data_dir, 'groups'),
            os.path.join(self.data_dir, 'final_signals')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        self.logger.info(f"Created/verified directories: {directories}")

    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object."""
        if pd.isna(timestamp_str) or timestamp_str == '':
            return None
            
        # Handle different timestamp formats
        timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S'
        ]
        
        for fmt in timestamp_formats:
            try:
                return datetime.strptime(str(timestamp_str), fmt)
            except ValueError:
                continue
        
        # If all formats fail, try pandas to_datetime
        try:
            return pd.to_datetime(timestamp_str)
        except:
            self.logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None

    def time_diff_minutes(self, timestamp1, timestamp2):
        """Calculate time difference in minutes between two timestamps."""
        if timestamp1 is None or timestamp2 is None:
            return float('inf')  # Return large number if can't calculate
        
        try:
            dt1 = self.parse_timestamp(timestamp1) if isinstance(timestamp1, str) else timestamp1
            dt2 = self.parse_timestamp(timestamp2) if isinstance(timestamp2, str) else timestamp2
            
            if dt1 is None or dt2 is None:
                return float('inf')
            
            diff = abs((dt2 - dt1).total_seconds() / 60.0)
            return diff
        except Exception as e:
            self.logger.warning(f"Error calculating time difference: {e}")
            return float('inf')

    def has_signal_keyword(self, text: str) -> bool:
        """Check if text contains any signal keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in SIGNAL_KEYWORDS)

    def has_valid_symbol(self, text: str) -> bool:
        """Check if text contains any valid trading symbols."""
        text_upper = text.upper()
        return any(symbol in text_upper for symbol in VALID_SYMBOLS)
    
    def is_signal_trigger(self, text: str) -> bool:
        """Check if text contains signal keywords or valid symbols."""
        return self.has_signal_keyword(text) or self.has_valid_symbol(text)

    def create_time_based_windows(self, messages: List[Dict]) -> List[Dict]:
        """Create time-based windows around signal messages."""
        windows = []
        processed_indices = set()  # Keep track of processed message indices
        
        self.logger.info(f"Creating time-based windows from {len(messages)} messages with time_window={self.time_window_minutes} minutes")
        
        for i, msg in enumerate(messages):
            # Skip if already processed
            if i in processed_indices:
                continue
                
            text = msg.get('text', '').strip()
            
            # Skip empty messages
            if not text:
                continue
            
            # Check if this message triggers a signal window
            if self.is_signal_trigger(text):
                window_messages = [msg]
                window_indices = {i}
                start_time = msg.get('timestamp', msg.get('date', ''))
                
                # Look ahead for messages within the time window
                j = i + 1
                while j < len(messages):
                    next_msg = messages[j]
                    next_text = next_msg.get('text', '').strip()
                    next_timestamp = next_msg.get('timestamp', next_msg.get('date', ''))
                    
                    # Skip empty messages
                    if not next_text:
                        j += 1
                        continue
                    
                    # Calculate time difference
                    time_diff = self.time_diff_minutes(start_time, next_timestamp)
                    
                    # If within time window, add to current window
                    if time_diff <= self.time_window_minutes:
                        window_messages.append(next_msg)
                        window_indices.add(j)
                        j += 1
                    else:
                        # Outside time window, stop expanding this window
                        break
                
                # Only create window if it has meaningful content
                if len(window_messages) > 0:
                    # Mark all indices in this window as processed
                    processed_indices.update(window_indices)
                    
                    # Determine signal quality
                    combined_text = ' '.join(m['text'] for m in window_messages if m['text'].strip())
                    signal_quality = self.assess_signal_quality(combined_text)
                    
                    # Create the window
                    window = {
                        'messages': window_messages,
                        'start_time': start_time,
                        'end_time': window_messages[-1].get('timestamp', window_messages[-1].get('date', '')),
                        'window_id': len(windows),
                        'trigger_message': text,
                        'trigger_type': 'time_based_signal',
                        'signal_quality': signal_quality,
                        'message_position': i,
                        'time_span_minutes': self.time_diff_minutes(start_time, window_messages[-1].get('timestamp', window_messages[-1].get('date', ''))),
                        'num_messages': len(window_messages)
                    }
                    windows.append(window)
                    
                    self.logger.debug(f"Created {signal_quality} quality window {len(windows)-1} starting at message {i}, spanning {window['time_span_minutes']:.2f} minutes with {len(window_messages)} messages")
        
        self.logger.info(f"Created {len(windows)} time-based windows")
        
        # Log window quality distribution
        quality_counts = {}
        for window in windows:
            quality = window.get('signal_quality', 'unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        self.logger.info(f"Window quality distribution: {quality_counts}")
        
        return windows

    def assess_signal_quality(self, combined_text: str) -> str:
        """Assess the quality of a signal based on its content."""
        text_lower = combined_text.lower()
        text_upper = combined_text.upper()
        
        # Check for symbol presence
        has_symbol = self.has_valid_symbol(combined_text)
        
        # Check for direction keywords
        direction_keywords = ['buy', 'sell', 'long', 'short']
        has_direction = any(keyword in text_lower for keyword in direction_keywords)
        
        # Check for price-related keywords
        price_keywords = ['tp', 'sl', 'target', 'stop', 'entry', 'price']
        has_price_info = any(keyword in text_lower for keyword in price_keywords)
        
        # Check for numeric values (potential prices)
        has_numbers = bool(re.search(r'\d+', combined_text))
        
        # Assess quality
        if has_symbol and has_direction and has_price_info and has_numbers:
            return 'high'
        elif has_symbol and (has_direction or (has_price_info and has_numbers)):
            return 'medium'
        else:
            return 'low'

    def process_group(self, group_name: str) -> bool:
        """Process a single group."""
        self.logger.info(f"Processing group: {group_name}")
        
        group_dir = os.path.join(self.data_dir, 'groups', group_name)
        raw_path = os.path.join(group_dir, 'raw_messages.csv')
        windows_path = os.path.join(group_dir, 'windows.csv')

        # Check if raw messages file exists
        if not os.path.exists(raw_path):
            self.logger.warning(f"Raw messages file not found: {raw_path}")
            return False

        try:
            # Read raw messages
            df = pd.read_csv(raw_path)
            self.logger.info(f"Loaded {len(df)} messages from {raw_path}")
            
            # Check available columns
            self.logger.debug(f"Available columns: {df.columns.tolist()}")

            # Determine which column to use for message text
            message_col = None
            for col in ['text', 'message', 'content']:
                if col in df.columns:
                    message_col = col
                    break
            
            if not message_col:
                self.logger.error(f"No text column found in {raw_path}. Available columns: {df.columns.tolist()}")
                return False

            # Determine which column to use for timestamp
            timestamp_col = None
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if not timestamp_col:
                self.logger.error(f"No timestamp column found in {raw_path}. Available columns: {df.columns.tolist()}")
                return False

            # Sort by timestamp to ensure chronological order
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)

            # Convert to list of message dictionaries
            messages = []
            for idx, row in df.iterrows():
                message = {
                    'text': str(row[message_col]) if pd.notna(row[message_col]) else '',
                    'message_id': row.get('message_id', idx),
                    'timestamp': row[timestamp_col],
                    'date': row[timestamp_col]  # For backward compatibility
                }
                messages.append(message)

            self.logger.info(f"Converted {len(messages)} messages to dictionary format")

            # Create time-based windows
            windows = self.create_time_based_windows(messages)

            # Convert windows to DataFrame
            windows_data = []
            for window in windows:
                combined_text = '\n'.join(msg['text'] for msg in window['messages'] if msg['text'].strip())
                windows_data.append({
                    'window_id': window['window_id'],
                    'combined_text': combined_text,
                    'trigger_message': window['trigger_message'],
                    'trigger_type': window['trigger_type'],
                    'signal_quality': window.get('signal_quality', 'unknown'),
                    'message_position': window.get('message_position', -1),
                    'start_message_id': window['messages'][0]['message_id'],
                    'end_message_id': window['messages'][-1]['message_id'],
                    'start_date': window['start_time'],
                    'end_date': window['end_time'],
                    'time_span_minutes': window.get('time_span_minutes', 0),
                    'num_messages': len(window['messages'])
                })

            # Save windows
            if windows_data:
                windows_df = pd.DataFrame(windows_data)
                windows_df.to_csv(windows_path, index=False)
                self.logger.info(f"Saved {len(windows_df)} windows to {windows_path}")
            else:
                self.logger.info(f"No windows created for group {group_name}")
                # Create empty windows file
                empty_df = pd.DataFrame(columns=[
                    'window_id', 'combined_text', 'trigger_message', 'trigger_type', 'signal_quality',
                    'message_position', 'start_message_id', 'end_message_id', 'start_date', 'end_date', 
                    'time_span_minutes', 'num_messages'
                ])
                empty_df.to_csv(windows_path, index=False)

            return True

        except Exception as e:
            self.logger.error(f"Error processing group {group_name}: {e}")
            return False

    def process_all_groups(self):
        """Process all groups specified in the configuration."""
        self.logger.info(f"Starting to process {len(self.groups)} groups")
        
        groups_dir = os.path.join(self.data_dir, 'groups')
        if not os.path.exists(groups_dir):
            self.logger.error(f"Groups directory not found: {groups_dir}")
            return

        successful_groups = 0
        failed_groups = 0

        for group_name in self.groups:
            group_dir = os.path.join(groups_dir, group_name)
            if not os.path.isdir(group_dir):
                self.logger.warning(f"Group directory not found: {group_dir}. Skipping.")
                failed_groups += 1
                continue

            if self.process_group(group_name):
                successful_groups += 1
            else:
                failed_groups += 1

        self.logger.info(f"Processing complete. Success: {successful_groups}, Failed: {failed_groups}")
            
def main():
    """Main function to run the signal extractor."""
    try:
        extractor = SignalExtractor()
        extractor.process_all_groups()
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main() 