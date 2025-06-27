import pandas as pd
import logging
import ta

class Indicators:
    def __init__(self, df):
        self.df = df

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            if df is None or df.empty:
                logging.error("No data to calculate indicators")
                return None


            required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"Missing required column: {col}")

            df['volume'] = df['tick_volume']  # Ensure 'volume' column is set correctly

            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['Signal'] = macd.macd_signal()
            stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['Stochastic'] = stochastic.stoch()
            df['Williams_%R'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
            df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['Ultimate_Oscillator'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['Ichimoku_Conversion_Line'] = ichimoku.ichimoku_conversion_line()
            df['Ichimoku_Base_Line'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            df['Ichimoku_Span_B'] = ichimoku.ichimoku_b()

            bollinger = ta.volatility.BollingerBands(df['close'])
            df['Bollinger_Middle'] = bollinger.bollinger_mavg()
            df['Bollinger_Upper'] = bollinger.bollinger_hband()
            df['Bollinger_Lower'] = bollinger.bollinger_lband()

            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['Keltner_Middle'] = keltner.keltner_channel_mband()
            df['Keltner_Upper'] = keltner.keltner_channel_hband()
            df['Keltner_Lower'] = keltner.keltner_channel_lband()

            donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            df['Donchian_Middle'] = donchian.donchian_channel_mband()
            df['Donchian_Upper'] = donchian.donchian_channel_hband()
            df['Donchian_Lower'] = donchian.donchian_channel_lband()

            df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
            df['HMA'] = ta.trend.WMAIndicator(df['close'], window=9).wma()
            df['EMV'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()
            df['ROC'] = ta.momentum.ROCIndicator(df['close']).roc()
            df['Bull_Power'] = df['high'] - ta.trend.EMAIndicator(df['close']).ema_indicator()
            df['Bear_Power'] = df['low'] - ta.trend.EMAIndicator(df['close']).ema_indicator()

            df.ffill(inplace=True)
            df.dropna(inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise
