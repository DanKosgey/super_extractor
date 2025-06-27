import os
import logging
from datetime import datetime
import pandas as pd
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from signal_extractor import SignalExtractor
from signal_parser import SignalParser

class TestPipeline:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        self.extractor = SignalExtractor(config_path)
        self.parser = SignalParser(config_path)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'test_pipeline.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run(self, user_input):
        self.logger.info("Starting test pipeline...")
        self.logger.info("Step 1: Simulating DataFrame with user input...")
        # Simulate a DataFrame as if it were loaded from forex_raw_signals.csv
        df = pd.DataFrame([
            {
                'message': user_input,
                'timestamp': datetime.now().isoformat(),
                'message_id': 0
            }
        ])
        # Use the extractor's filter and combine logic on this df
        filtered_df = self.extractor.filter_messages(df)
        combined_messages = self.extractor.combine_messages(filtered_df)
        self.logger.info(f"Combined {len(combined_messages)} messages.")

        self.logger.info("Step 2: Saving messages...")
        self.extractor.save_prediction_messages(combined_messages)
        self.logger.info("Messages saved.")

        self.logger.info("Step 3: Parsing and validating signals...")
        signals = self.parser.parse_signals(combined_messages)
        self.logger.info(f"Parsed {len(signals)} signals.")

        self.logger.info("Step 4: Making predictions...")
        predictions = self.parser.make_predictions(signals)
        self.logger.info(f"Made {len(predictions)} predictions.")

        self.logger.info("Step 5: Saving final results...")
        self.parser.save_predictions(predictions)
        self.logger.info("Results saved.")

        return predictions

def main():
    pipeline = TestPipeline()
    user_input = input("Enter a signal (e.g., 'BUY EURGBP 1.3 1.9869 [1.292, 0.85, 1.808]'): ")
    predictions = pipeline.run(user_input)
    if predictions:
        print("\nModel Predictions:")
        for prediction in predictions:
            print(f"\nSignal: {prediction['direction']} {prediction['symbol']}")
            print(f"Entry Price: {prediction['entry_price']}")
            print(f"Stop Loss: {prediction['stop_loss']}")
            print(f"Take Profits: {prediction['take_profits']}")
            print(f"Model Analysis: {prediction['model_analysis']}")
    else:
        print("Pipeline failed. Check the logs for details.")

if __name__ == "__main__":
    main() 