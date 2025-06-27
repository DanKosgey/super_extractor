import os
import pandas as pd
import yaml
import logging
from typing import Dict, List, Tuple, Optional
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
        
        # Get data directory from config, fallback to default if not present
        self.data_dir = self.config.get('data_dir') or self.config.get('signal_extraction', {}).get('data_dir')
        if not self.data_dir:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.data_dir = os.path.abspath(self.data_dir)
        self.logger.info(f"Using data directory: {self.data_dir}")
        
        # Create necessary directories
        self.create_directories()
        
        # Get time window from config (in minutes) - this is the 't' parameter you mentioned
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
        """Setup logging to console only."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, 'groups')
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

    def detect_signal_components(self, text: str) -> Dict[str, bool]:
        """Detect specific signal components in text."""
        text_lower = text.lower()
        
        # Check for SL (Stop Loss)
        has_sl = any(keyword in text_lower for keyword in ['sl', 'stop loss', 'stoploss', 'stop'])
        
        # Check for TP (Take Profit)
        has_tp = any(keyword in text_lower for keyword in ['tp', 'take profit', 'takeprofit', 'target'])
        
        # Check for Direction
        direction_keywords = ['buy', 'sell', 'long', 'short', 'bullish', 'bearish']
        has_direction = any(keyword in text_lower for keyword in direction_keywords)
        
        return {
            'has_sl': has_sl,
            'has_tp': has_tp,
            'has_direction': has_direction
        }

    def is_complete_signal(self, text: str) -> bool:
        """Check if text contains all three components: SL, TP, and Direction."""
        components = self.detect_signal_components(text)
        return components['has_sl'] and components['has_tp'] and components['has_direction']

    def has_partial_signal(self, text: str) -> bool:
        """Check if text contains TP and SL, or Direction (but not all three)."""
        components = self.detect_signal_components(text)
        
        # Has TP and SL but not direction
        tp_sl_only = components['has_tp'] and components['has_sl'] and not components['has_direction']
        
        # Has direction but not both TP and SL
        direction_only = components['has_direction'] and not (components['has_tp'] and components['has_sl'])
        
        return tp_sl_only or direction_only

    def create_perfect_time_windows(self, messages: List[Dict]) -> List[Dict]:
        """Create perfect time-based windows based on signal completeness."""
        windows = []
        processed_indices = set()
        
        self.logger.info(f"Creating perfect time windows from {len(messages)} messages with t={self.time_window_minutes} minutes")
        
        for i, msg in enumerate(messages):
            # Skip if already processed
            if i in processed_indices:
                continue
                
            text = msg.get('text', '').strip()
            
            # Skip empty messages
            if not text:
                continue
            
            # Check if this message is a signal trigger
            if not self.is_signal_trigger(text):
                continue
                
            trigger_timestamp = self.parse_timestamp(msg.get('timestamp', msg.get('date', '')))
            if trigger_timestamp is None:
                continue
                
            # Determine window type based on signal completeness
            if self.is_complete_signal(text):
                # Complete signal: save as single-message window
                window = self.create_single_message_window(msg, i, len(windows))
                windows.append(window)
                processed_indices.add(i)
                self.logger.debug(f"Created complete signal window at message {i}")
                
            elif self.has_partial_signal(text) or self.is_signal_trigger(text):
                # Partial signal or general trigger: create extended window
                window, used_indices = self.create_extended_window(messages, i, trigger_timestamp, len(windows))
                if window:
                    windows.append(window)
                    processed_indices.update(used_indices)
                    self.logger.debug(f"Created extended window at message {i}, spanning {len(used_indices)} messages")
        
        self.logger.info(f"Created {len(windows)} perfect time windows")
        return windows

    def create_single_message_window(self, msg: Dict, msg_index: int, window_id: int) -> Dict:
        """Create a window for a single complete signal message."""
        timestamp = msg.get('timestamp', msg.get('date', ''))
        
        return {
            'messages': [msg],
            'start_time': timestamp,
            'end_time': timestamp,
            'window_id': window_id,
            'trigger_message': msg.get('text', ''),
            'trigger_type': 'complete_signal',
            'signal_quality': 'complete',
            'message_position': msg_index,
            'time_span_minutes': 0,
            'num_messages': 1,
            'trigger_timestamp': timestamp
        }

    def create_extended_window(self, messages: List[Dict], trigger_index: int, trigger_timestamp: datetime, window_id: int) -> Tuple[Optional[Dict], set]:
        """Create an extended window starting from trigger message + t minutes."""
        window_messages = [messages[trigger_index]]
        used_indices = {trigger_index}
        
        # Calculate the end time for the window (trigger + t minutes)
        window_end_time = trigger_timestamp + timedelta(minutes=self.time_window_minutes)
        
        # Collect all messages within the time window
        for j in range(trigger_index + 1, len(messages)):
            next_msg = messages[j]
            next_text = next_msg.get('text', '').strip()
            
            # Skip empty messages
            if not next_text:
                continue
                
            next_timestamp = self.parse_timestamp(next_msg.get('timestamp', next_msg.get('date', '')))
            
            if next_timestamp is None:
                continue
                
            # Check if message is within the time window
            if next_timestamp <= window_end_time:
                window_messages.append(next_msg)
                used_indices.add(j)
            else:
                # Outside time window, stop collecting
                break
        
        # Only create window if we have messages
        if len(window_messages) == 0:
            return None, set()
            
        # Determine signal quality for the entire window
        combined_text = ' '.join(m.get('text', '') for m in window_messages if m.get('text', '').strip())
        signal_quality = self.assess_signal_quality(combined_text)
        
        # Calculate actual time span
        last_timestamp = self.parse_timestamp(window_messages[-1].get('timestamp', window_messages[-1].get('date', '')))
        actual_time_span = 0
        if last_timestamp and trigger_timestamp:
            actual_time_span = (last_timestamp - trigger_timestamp).total_seconds() / 60.0
        
        window = {
            'messages': window_messages,
            'start_time': messages[trigger_index].get('timestamp', messages[trigger_index].get('date', '')),
            'end_time': window_messages[-1].get('timestamp', window_messages[-1].get('date', '')),
            'window_id': window_id,
            'trigger_message': messages[trigger_index].get('text', ''),
            'trigger_type': 'extended_signal',
            'signal_quality': signal_quality,
            'message_position': trigger_index,
            'time_span_minutes': actual_time_span,
            'num_messages': len(window_messages),
            'trigger_timestamp': messages[trigger_index].get('timestamp', messages[trigger_index].get('date', '')),
            'planned_window_minutes': self.time_window_minutes
        }
        
        return window, used_indices

    def assess_signal_quality(self, combined_text: str) -> str:
        """Assess the quality of a signal based on its content."""
        text_lower = combined_text.lower()
        
        # Check for components
        components = self.detect_signal_components(combined_text)
        has_symbol = self.has_valid_symbol(combined_text)
        has_numbers = bool(re.search(r'\d+', combined_text))
        
        # Assess quality based on completeness
        if self.is_complete_signal(combined_text) and has_symbol and has_numbers:
            return 'complete'
        elif components['has_sl'] and components['has_tp'] and has_symbol:
            return 'high'
        elif (components['has_direction'] and has_symbol) or (components['has_tp'] or components['has_sl']):
            return 'medium'
        elif has_symbol or any(components.values()):
            return 'low'
        else:
            return 'minimal'

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

            # Create perfect time-based windows
            windows = self.create_perfect_time_windows(messages)

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
                    'num_messages': len(window['messages']),
                    'trigger_timestamp': window.get('trigger_timestamp', ''),
                    'planned_window_minutes': window.get('planned_window_minutes', self.time_window_minutes)
                })

            # Save windows
            if windows_data:
                windows_df = pd.DataFrame(windows_data)
                windows_df.to_csv(windows_path, index=False)
                self.logger.info(f"Saved {len(windows_df)} windows to {windows_path}")
                
                # Log statistics
                quality_counts = windows_df['signal_quality'].value_counts()
                trigger_type_counts = windows_df['trigger_type'].value_counts()
                self.logger.info(f"Signal quality distribution: {quality_counts.to_dict()}")
                self.logger.info(f"Trigger type distribution: {trigger_type_counts.to_dict()}")
                
            else:
                self.logger.info(f"No windows created for group {group_name}")
                # Create empty windows file
                empty_df = pd.DataFrame(columns=[
                    'window_id', 'combined_text', 'trigger_message', 'trigger_type', 'signal_quality',
                    'message_position', 'start_message_id', 'end_message_id', 'start_date', 'end_date', 
                    'time_span_minutes', 'num_messages', 'trigger_timestamp', 'planned_window_minutes'
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