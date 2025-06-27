import logging  # Importing the logging module to log messages, errors, and debugging information.
import pandas as pd  # Importing pandas for data manipulation, particularly working with DataFrames.

# Defining the FeatureEngineering class to calculate feature changes for financial data.
class FeatureEngineering:
    def __init__(self, df):  # Constructor method to initialize the FeatureEngineering class.
        self.df = df  # Store the DataFrame (df) as an instance variable for use in class methods.

    def calculate_changes(self, df: pd.DataFrame):
        if df is None:  # If the DataFrame is None (empty or invalid), log an error message.
            logging.error("No data to calculate changes")  # Log an error indicating the lack of data.
            return None  # Return None to signal that no processing can be done due to missing data.
        
        try:
            # Remove timestamp columns if present in the DataFrame.
            if 'time' in df.columns:  # Check if the 'time' column exists in the DataFrame.
                df = df.drop(columns=['time'])  # Drop the 'time' column as it is not needed for feature engineering.

            # Calculate the change in tick volume between consecutive rows.
            df['Volume_Change'] = df['tick_volume'].diff()  # Create a new column 'Volume_Change' to store the difference in tick volume.

            # Calculate the change in the RSI (Relative Strength Index) between consecutive rows.
            df['RSI_Change'] = df['RSI'].diff()  # Create a new column 'RSI_Change' to store the difference in RSI values.

            # Calculate the change in the Rate of Change (ROC) between consecutive rows.
            df['ROC_Change'] = df['ROC'].diff()  # Create a new column 'ROC_Change' to store the difference in ROC values.

            # Calculate the change in close price between consecutive rows.
            df['close_change'] = df['close'].diff()  # Create a new column 'close_change' to store the difference in close price.

            # Create new columns to represent the direction of price changes.
            # These columns will store 1 if the price increased, and 0 if it decreased.
            df['close_dir'] = (df['close'].diff() > 0).astype(int)  # 1 if the close price increased, otherwise 0.
            df['high_dir'] = (df['high'].diff() > 0).astype(int)  # 1 if the high price increased, otherwise 0.
            df['low_dir'] = (df['low'].diff() > 0).astype(int)  # 1 if the low price increased, otherwise 0.
            df['open_dir'] = (df['open'].diff() > 0).astype(int)  # 1 if the open price increased, otherwise 0.

            # Fill any missing data (NaN) with 0 to prevent errors during further calculations.
            df.fillna(0, inplace=True)  # Replace all NaN values in the DataFrame with 0.
            
            # Drop any remaining rows that still contain NaN values.
            df.dropna(inplace=True)  # Remove rows that still contain NaN values after filling.

            # Sort the DataFrame by index to ensure the data is in chronological order.
            df.sort_index(inplace=True)  # Sort the DataFrame in place by its index (typically time-based).

            # Drop specific columns that are no longer needed after feature engineering.
            df.drop(columns=[ 'high', 'low', 'open', 'real_volume','spread'], inplace=True)  # Drop columns not needed for further analysis.

            # Return the DataFrame with the calculated feature changes and new direction columns.
            return df  # Return the modified DataFrame with the new features.

        except Exception as e:  # Catch any exceptions that occur during the feature engineering process.
            logging.error(f"Error calculating changes: {e}")  # Log the error message with exception details.
            raise  # Reraise the exception to propagate the error and halt further processing if necessary.
