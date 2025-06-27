#!/usr/bin/env python3
"""
Gemini Signal Parser
Extracts and validates trading signals from Telegram messages using Gemini AI.
Enhanced with dynamic file creation and robust progress tracking.
"""

import os
import sys
import csv
import json
import time
import yaml
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, cast
from dataclasses import dataclass, asdict
from pathlib import Path
import google.generativeai as genai  # type: ignore
import re

# Import the progress tracker
from progress_tracker import ProgressTracker


@dataclass
class TradingSignal:
    """Represents a parsed trading signal"""
    group_name: str
    window_id: str
    timestamp: str
    symbol: str
    direction: str
    entry: float
    sl: float  # stop loss
    tp1: float  # take profit 1
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    confidence: float = 0.0
    valid: bool = True
    corrections: str = ""
    validation_notes: str = ""  # Initialize as an empty string
    context: str = ""
    raw_text: str = ""


class SignalParser:
    """Main signal extraction class using Gemini AI with enhanced functionality"""
    
    def __init__(self, config_path: str):
        """Initialize the SignalParser with configuration and setup necessary components"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        self.setup_gemini()
        
        # Initialize progress tracker with proper path
        progress_file = Path(self.config['paths']['data_dir']) / "processing_state.json"
        self.progress_tracker = ProgressTracker(str(progress_file))
        
        # Initialize rate limiter
        from rate_limiter import GeminiRateLimiter
        self.rate_limiter = GeminiRateLimiter(self.config)
        self.rate_limiter.start()

        # Load SL/TP patterns from config
        self.load_patterns_from_config()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Ensure required paths exist in config
            required_paths = ['data_dir', 'logs_dir', 'final_signals_file']
            for path in required_paths:
                if path not in config.get('paths', {}):
                    raise ValueError(f"Missing required path in config: {path}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def setup_directories(self):
        """Ensure all required directories exist"""
        paths_to_create = [
            self.config['paths']['data_dir'],
            self.config['paths']['logs_dir'],
            Path(self.config['paths']['final_signals_file']).parent,
        ]
        
        # Create directories
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {path}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging_config = self.config.get('logging', {})
        self.logger = logging.getLogger('SignalParser')
        self.logger.setLevel(logging_config.get('level', 'INFO'))

        # Remove all handlers associated with the logger object (avoid duplicate logs)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Setup file handler (force logs/signal_parser.log)
        log_file = 'logs/signal_parser.log'
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Test log entry to confirm file logging
        self.logger.info('Logger initialized and writing to logs/signal_parser.log')
        
    def setup_gemini(self):
        """Setup Gemini API with configuration"""
        try:
            # Always get API key from config.yaml
            api_key = self.config['gemini']['api_key']
            if not api_key:
                raise Exception("No Gemini API key found in config.yaml")
            genai.configure(api_key=api_key)

            # Use model from config, fallback to gemini-1.5-flash
            model_name = self.config['gemini'].get('model', 'models/gemini-1.5-flash')
            self.model = genai.GenerativeModel(model_name)

            self.generation_params = {
                'temperature': self.config['gemini'].get('temperature', 0.2),
                'top_p': self.config['gemini'].get('top_p', 0.8),
                'top_k': self.config['gemini'].get('top_k', 40),
                'max_output_tokens': self.config['gemini'].get('max_tokens', 1024),
                'candidate_count': 1
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
    
    def rate_limit_check(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # Reset counter every minute
        if time_diff > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If we've made too many requests, wait
        max_requests = self.config['gemini'].get('max_requests_per_minute', 60)
        if self.request_count >= max_requests:
            sleep_time = 60 - time_diff
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
    
    def contains_required_keywords(self, text: str) -> bool:
        """
        Check if text contains required trading signal keywords.
        Must contain: (tp OR tp1) AND direction (buy/sell/long/short) AND sl (stop loss)
        """
        text_lower = text.lower()
        
        # Define keyword patterns
        tp_pattern = r'\b(?:tp|tp1|take profit|takeprofit)\b'
        direction_pattern = r'\b(?:buy|sell|long|short)\b'
        sl_pattern = r'\b(?:sl|stop loss|stoploss)\b'
        
        # Check for all required patterns
        has_tp = bool(re.search(tp_pattern, text_lower))
        has_direction = bool(re.search(direction_pattern, text_lower))
        has_sl = bool(re.search(sl_pattern, text_lower))
        
        return has_tp and has_direction and has_sl
    
    def build_extraction_prompt(self, text: str) -> str:
        """Build the prompt for Gemini to extract trading signals"""
        valid_symbols = ", ".join(self.config['signal_extraction']['valid_symbols'])
        
        prompt = f"""
Analyze the following trading signal text and extract trading information. Return ONLY a valid JSON object with the following structure:

{{
    "symbol": "SYMBOL_NAME",
    "direction": "buy" or "sell",
    "entry": entry_price_as_number,
    "sl": stop_loss_price_as_number,
    "tp1": take_profit_1_as_number,
    "tp2": take_profit_2_as_number_or_null,
    "tp3": take_profit_3_as_number_or_null,
    "tp4": take_profit_4_as_number_or_null,
    "confidence": confidence_score_0_to_1,
    "corrections": "any_corrections_made",
    "validation_notes": "validation_comments",
    "context": "additional_context_found"
}}

Rules:
1. Valid symbols: {valid_symbols}
2. Direction must be "buy" or "sell" (convert long→buy, short→sell)
3. All prices must be valid numbers
4. For BUY signals: sl < entry < tp1 < tp2 < tp3 < tp4
5. For SELL signals: sl > entry > tp1 > tp2 > tp3 > tp4
6. If signal is invalid or unclear, set confidence to 0
7. Correct common typos (EUUSD → EURUSD, etc.)
8. Extract all available take profit levels
9. If no clear signal found, return {{"confidence": 0}}

Text to analyze:
{text}

Return only the JSON object, no other text:"""
        
        return prompt
    
    def extract_signal_with_gemini(self, text: str, window_id: str, group_name: str, retries: int = 2) -> Optional[TradingSignal]:
        """Extract trading signal using Gemini AI with rate limiting and retry logic"""
        if not self.contains_required_keywords(text):
            self.logger.debug(f"Skipping window {window_id} - missing required keywords")
            return None
            
        prompt = self.build_extraction_prompt(text)
        last_exception = None
        for attempt in range(retries + 1):
            try:
                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    self.logger.warning(f"Empty response from Gemini for window {window_id}")
                    continue
                clean_response = str(response.text).strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                result = json.loads(clean_response.strip())
                if result.get('confidence', 0) == 0:
                    self.logger.info(f"Signal extraction failed for window {window_id}: low confidence")
                    return None
                signal = TradingSignal(
                    group_name=group_name,
                    window_id=window_id,
                    timestamp=datetime.now().isoformat(),
                    symbol=result['symbol'],
                    direction=result['direction'],
                    entry=float(result['entry']),
                    sl=float(result['sl']),
                    tp1=float(result['tp1']),
                    tp2=float(result.get('tp2')) if result.get('tp2') is not None else None,
                    tp3=float(result.get('tp3')) if result.get('tp3') is not None else None,
                    tp4=float(result.get('tp4')) if result.get('tp4') is not None else None,
                    confidence=float(result.get('confidence', 0.0)),
                    corrections=result.get('corrections', ''),
                    validation_notes=result.get('validation_notes', ''),
                    context=result.get('context', ''),
                    raw_text=text
                )
                signal.valid = self.validate_signal(signal)
                return signal
            except Exception as e:
                last_exception = e
                if attempt < retries:
                    self.logger.warning(f"Retrying Gemini API (attempt {attempt+1}) for window {window_id} due to error: {e}")
                    time.sleep(2)
                else:
                    self.logger.error(f"Gemini API error after {retries+1} attempts for window {window_id}: {e}")
        return None
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal"""
        try:
            # Check required fields
            if not all([signal.symbol, signal.direction, signal.entry, signal.sl, signal.tp1]):
                signal.validation_notes += "Missing required fields. "
                return False
            
            # Check valid symbol
            if signal.symbol not in self.config['signal_extraction']['valid_symbols']:
                signal.validation_notes += f"Invalid symbol: {signal.symbol}. "
                return False
            
            # Check valid direction
            if signal.direction.lower() not in ['buy', 'sell']:
                signal.validation_notes += f"Invalid direction: {signal.direction}. "
                return False
            
            # Check price relationships
            if signal.direction.lower() == 'buy':
                if not (signal.sl < signal.entry < signal.tp1):
                    signal.validation_notes += "Invalid price structure for BUY signal. "
                    return False
                # Check TP progression
                if signal.tp2 and signal.tp2 <= signal.tp1:
                    signal.validation_notes += "TP2 must be greater than TP1. "
                    return False
                if signal.tp3 and (not signal.tp2 or signal.tp3 <= signal.tp2):
                    signal.validation_notes += "TP3 must be greater than TP2. "
                    return False
                if signal.tp4 and (not signal.tp3 or signal.tp4 <= signal.tp3):
                    signal.validation_notes += "TP4 must be greater than TP3. "
                    return False
            
            elif signal.direction.lower() == 'sell':
                if not (signal.sl > signal.entry > signal.tp1):
                    signal.validation_notes += "Invalid price structure for SELL signal. "
                    return False
                # Check TP progression
                if signal.tp2 and signal.tp2 >= signal.tp1:
                    signal.validation_notes += "TP2 must be less than TP1. "
                    return False
                if signal.tp3 and (not signal.tp2 or signal.tp3 >= signal.tp2):
                    signal.validation_notes += "TP3 must be less than TP2. "
                    return False
                if signal.tp4 and (not signal.tp3 or signal.tp4 >= signal.tp3):
                    signal.validation_notes += "TP4 must be less than TP3. "
                    return False
            
            signal.validation_notes += "Valid signal structure. "
            return True
            
        except Exception as e:
            signal.validation_notes += f"Validation error: {e}. "
            return False

    def load_windows_data(self, group_name: str) -> pd.DataFrame:
        """Load windows data for a group with error handling"""
        windows_path = Path(self.config['paths']['windows_dir'].format(group_name=group_name))
        
        if not windows_path.exists():
            self.logger.error(f"Windows file not found: {windows_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(windows_path, encoding='utf-8')
            self.logger.info(f"Loaded {len(df)} windows for group {group_name}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading windows data for {group_name}: {e}")
            return pd.DataFrame()
    
    def create_signals_file_if_not_exists(self, file_path: Path):
        """Create signals CSV file with required columns if it doesn't exist"""
        if not file_path.exists():
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Define required columns based on TradingSignal dataclass
            columns = [
                'group_name', 'window_id', 'timestamp', 'symbol', 'direction',
                'entry', 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'confidence',
                'valid', 'corrections', 'validation_notes', 'context', 'raw_text'
            ]
            
            # Create empty DataFrame with columns and save
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(file_path, index=False, encoding='utf-8')
            self.logger.info(f"Created signals file with required columns: {file_path}")
    
    def save_signals(self, signals: list, group_name: str):
        """Save extracted signals (merged with window data) to CSV with dynamic file creation"""
        if not signals:
            return
        try:
            # Convert signals to DataFrame (signals are already dicts with merged window+signal fields)
            new_df = pd.DataFrame(signals)
            # Save to group-specific file
            group_file = Path(self.config['paths']['signals_dir'].format(group_name=group_name))
            group_file.parent.mkdir(parents=True, exist_ok=True)
            if group_file.exists():
                existing_df = pd.read_csv(group_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            combined_df.to_csv(group_file, index=False)
            # Save to combined signals file
            combined_file = Path(self.config['paths']['final_signals_file'])
            combined_file.parent.mkdir(parents=True, exist_ok=True)
            if combined_file.exists():
                new_df.to_csv(combined_file, mode='a', header=False, index=False)
            else:
                new_df.to_csv(combined_file, index=False)
            self.logger.info(f"Saved {len(signals)} signals for group {group_name}")
        except Exception as e:
            self.logger.error(f"Error saving signals for group {group_name}: {e}")
            raise
    
    def save_signal(self, signal: TradingSignal):
        """Save a trading signal to CSV"""
        try:
            # Get the output file path
            signals_dir_template = self.config['paths']['signals_dir']
            output_path = signals_dir_template.format(group_name=signal.group_name)
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert signal to dict
            signal_dict = asdict(signal)
            
            # Check if file exists to determine if we need headers
            write_header = not Path(output_path).exists()
            
            # Append to CSV
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(signal_dict.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(signal_dict)
                
            self.logger.debug(f"Saved signal for window {signal.window_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {str(e)}")
            raise

    def process_group(self, group_name: str, max_messages: Optional[int] = None):
        """Process all windows/messages for a specific group"""
        self.logger.info(f"Processing group: {group_name}")

        # Load windows data
        windows_df = self.load_windows_data(group_name)
        if max_messages:
            windows_df = windows_df.head(max_messages)
        if windows_df.empty:
            self.logger.warning(f"No windows data found for group {group_name}")
            return

        all_window_ids = list(map(str, windows_df['window_id'].tolist() if 'window_id' in windows_df.columns else windows_df.index.tolist()))
        # Determine resume index and first unprocessed window
        resume_idx = None
        for idx, window_id in enumerate(all_window_ids):
            if not self.progress_tracker.is_window_processed(group_name, window_id):
                resume_idx = idx
                break
        if resume_idx is None:
            self.logger.info(f"All windows already processed for group {group_name}. Nothing to do.")
            return
        self.logger.info(f"Resuming group '{group_name}' from window index {resume_idx} (window_id={all_window_ids[resume_idx]}) out of {len(all_window_ids)} windows. Skipping {resume_idx} already-processed windows.")
        # Check if group is already completed (all windows processed)
        if self.progress_tracker.is_group_completed(group_name, all_window_ids):
            if self.config['progress'].get('skip_completed_groups', True):
                self.logger.info(f"Skipping completed group: {group_name}")
                return
        batch_size = self.config['progress'].get('batch_size', 5)
        batch_signals = []
        processed_count = 0
        skipped_count = 0

        for idx, row in enumerate(windows_df.itertuples(index=False, name=None)):
            if idx < resume_idx:
                continue
            # row is a tuple, so get columns from windows_df.columns
            row_dict = dict(zip(windows_df.columns, row))
            window_id = str(row_dict.get('window_id', f"window_{idx}"))
            # Skip if already processed
            if self.progress_tracker.is_window_processed(group_name, window_id):
                skipped_count += 1
                continue
            # Use the best available text field
            window_text = str(row_dict.get('combined_text') or row_dict.get('text') or '')
            if not self.contains_required_keywords(window_text):
                skipped_count += 1
                self.logger.debug(f"Skipping window {window_id} - missing required keywords")
                continue
            self.logger.debug(f"Processing window {window_id} for group {group_name}")
            signal = self.extract_signal_with_gemini(window_text, window_id, group_name)
            if signal:
                if signal.valid:
                    # Merge window row and signal dict
                    merged = row_dict.copy()
                    merged.update(asdict(signal))
                    batch_signals.append(merged)
                    self.logger.info(f"Extracted valid signal for window {window_id}: {signal}")
                else:
                    skipped_count += 1
                    self.logger.warning(f"Extracted invalid signal for window {window_id}: {signal.validation_notes}")
                    if signal.validation_notes:
                        self.logger.debug(f"Validation notes for window {window_id}: {signal.validation_notes}")
            else:
                skipped_count += 1
                self.logger.debug(f"No valid signal extracted for window {window_id}")
                if signal and signal.validation_notes:
                    self.logger.debug(f"Validation notes for window {window_id}: {signal.validation_notes}")
                if signal and not signal.valid:
                    self.logger.error(f"Signal extraction failed for window {window_id}: {signal.validation_notes}")
                    if signal.validation_notes:
                        self.logger.debug(f"Validation notes for window {window_id}: {signal.validation_notes}")
                    else:
                        self.logger.debug(f"No validation notes for window {window_id}")
                    self.logger.debug(f"Signal extraction failed for window {window_id}: {signal.validation_notes}")


            # Update progress
            signal_count = 1 if signal else 0
            valid_count = 1 if signal and signal.valid else 0
            
            if hasattr(self.progress_tracker, 'update_group_progress'):
                self.progress_tracker.update_group_progress(group_name, window_id, signal_count, valid_count)
            
            # Mark window as processed
            self.progress_tracker.mark_completed(group_name, window_id)
            processed_count += 1

            # Save batch if size reached
            if len(batch_signals) >= batch_size:
                self.save_signals(batch_signals, group_name)
                batch_signals = []
                sleep_time = self.config['batching'].get('sleep_between_batches', 1)
                self.logger.info(f"Batch completed ({processed_count} processed, {skipped_count} skipped), sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

        # Save remaining signals
        if batch_signals:
            self.save_signals(batch_signals, group_name)

        # Mark group as completed
        if hasattr(self.progress_tracker, 'mark_group_completed'):
            self.progress_tracker.mark_group_completed(group_name)
        
        # Get stats
        stats = {}
        if hasattr(self.progress_tracker, 'get_group_stats'):
            stats = self.progress_tracker.get_group_stats(group_name)
        
        self.logger.info(
            f"Completed processing group {group_name} - "
            f"Processed: {processed_count}, Skipped: {skipped_count}, "
            f"Signals: {stats.get('total_signals', '?')}, Valid: {stats.get('valid_signals', '?')}"
        )

    def process_window_csv(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single window CSV file, saving after every batch of 12 requests and after sleep. Adds debug logging for skipped/failed rows."""
        # Use self.logger for class logging
        logger = logging.getLogger(__name__)
        try:
            df_window = pd.read_csv(csv_path)
            if 'combined_text' not in df_window.columns:
                raise ValueError("CSV must contain 'combined_text' column")
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        signals_rows = []
        skipped_rows = []
        error_rows = []
        window_name = Path(csv_path).parent.name
        total_rows = len(df_window)
        logger.info(f"Processing {total_rows} rows from {window_name}")
        batch_size = 12
        # Debug: print first 3 combined_texts
        for debug_idx in range(min(3, total_rows)):
            logger.debug(f"Sample combined_text [{debug_idx}]: {df_window.loc[debug_idx, 'combined_text']}")
        for idx in range(total_rows):
            try:
                text = str(df_window.loc[idx, 'combined_text'])
                has_sl, has_tp = self.has_sl_and_tp(text)
                if has_sl and has_tp:
                    logger.info(f"Processing index {idx} with Gemini AI")
                    parsed_signal = self.extract_signal_with_gemini(
                        text,
                        window_id=str(idx),
                        group_name=window_name
                    )
                    # Debug: log Gemini response
                    logger.debug(f"Gemini response for index {idx}: {parsed_signal}")
                    signal_row = {
                        "window": window_name,
                        "original_index": idx,
                        "combined_text": text,
                        **parsed_signal, # type: ignore
                        "processed_at": datetime.now().isoformat()
                    }
                    signals_rows.append(signal_row)
                elif has_sl or has_tp:
                    reason = f"Has {'SL' if has_sl else 'TP'} but missing {'TP' if has_sl else 'SL'}"
                    logger.debug(f"Skipped index {idx}: {reason} | Text: {text}")
                    skipped_rows.append({
                        "window": window_name,
                        "original_index": idx,
                        "combined_text": text,
                        "reason": reason,
                        "has_sl": has_sl,
                        "has_tp": has_tp
                    })
                else:
                    logger.debug(f"Skipped index {idx}: No SL or TP keywords found | Text: {text}")
                    skipped_rows.append({
                        "window": window_name,
                        "original_index": idx,
                        "combined_text": text,
                        "reason": "No SL or TP keywords found",
                        "has_sl": False,
                        "has_tp": False
                    })
            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")
                logger.debug(f"Failed text at index {idx}: {text if 'text' in locals() else 'N/A'}")
                error_rows.append({
                    "window": window_name,
                    "original_index": idx,
                    "combined_text": text if 'text' in locals() else "N/A",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            # Batch save after every 12 requests
            if (idx + 1) % batch_size == 0:
                logger.info(f"Batch save at index {idx}")
                self.save_window_results(csv_path, pd.DataFrame(signals_rows), pd.DataFrame(skipped_rows), pd.DataFrame(error_rows))
                signals_rows = []
                skipped_rows = []
                error_rows = [] 
            # Sleep after every batch to respect rate limits
            if (idx + 1) % batch_size == 0:
                sleep_time = self.config['batching'].get('sleep_between_batches', 1)
                logger.info(f"Sleeping for {sleep_time} seconds after processing {idx + 1} rows")
                time.sleep(sleep_time)
        # Save any remaining rows after the loop
        if signals_rows or skipped_rows or error_rows:
            logger.info(f"Final save after processing {total_rows} rows")
            self.save_window_results(csv_path, pd.DataFrame(signals_rows), pd.DataFrame(skipped_rows), pd.DataFrame(error_rows))
        # Final save after all rows
        self.save_window_results(csv_path, pd.DataFrame(signals_rows), pd.DataFrame(skipped_rows), pd.DataFrame(error_rows))
        df_signals = pd.DataFrame(signals_rows)
        df_skipped = pd.DataFrame(skipped_rows)
        df_errors = pd.DataFrame(error_rows)
        logger.info(f"Window {window_name} summary:")
        logger.info(f"  - Total rows: {total_rows}")
        logger.info(f"  - Signals extracted: {len(df_signals)}")
        logger.info(f"  - Rows skipped: {len(df_skipped)}")
        logger.info(f"  - Errors: {len(df_errors)}")
        return df_signals, df_skipped, df_errors

    def atomic_save_csv(self, df: pd.DataFrame, path: Path):
        """Atomically save a DataFrame to CSV using a temp file and move/replace."""
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    def save_summary_samples(self, window_dir: Path, df_skipped: pd.DataFrame, df_errors: pd.DataFrame, sample_size: int = 20):
        """Save a sample of skipped and error rows for quick review."""
        if not df_skipped.empty:
            skipped_sample = df_skipped.head(sample_size)
            summary_path = window_dir / "skipped_summary.csv"
            self.atomic_save_csv(skipped_sample, summary_path)
            self.logger.info(f"Saved skipped sample to: {summary_path}")
        if not df_errors.empty:
            errors_sample = df_errors.head(sample_size)
            summary_path = window_dir / "errors_summary.csv"
            self.atomic_save_csv(errors_sample, summary_path)
            self.logger.info(f"Saved error sample to: {summary_path}")

    def save_window_results(self, csv_path: str, df_signals: pd.DataFrame, df_skipped: pd.DataFrame, df_errors: pd.DataFrame):
        """Save results for a window in its directory using atomic writes and save summary samples."""
        window_dir = Path(csv_path).parent
        # Save signals
        if not df_signals.empty:
            signals_path = window_dir / "signals.csv"
            self.atomic_save_csv(df_signals, signals_path)
            self.logger.info(f"Saved signals to: {signals_path}")
        # Save skipped (for diagnostics)
        if not df_skipped.empty:
            skipped_path = window_dir / "skipped.csv"
            self.atomic_save_csv(df_skipped, skipped_path)
            self.logger.info(f"Saved skipped to: {skipped_path}")
        # Save errors (for debugging)
        if not df_errors.empty:
            errors_path = window_dir / "errors.csv"
            self.atomic_save_csv(df_errors, errors_path)
            self.logger.info(f"Saved errors to: {errors_path}")
        # Save summary samples
        self.save_summary_samples(window_dir, df_skipped, df_errors)

    def load_patterns_from_config(self):
        """Load SL/TP regex patterns from config, fallback to defaults if not present."""
        sl_patterns = self.config.get('patterns', {}).get('sl', [
            r'\\bsl\\b', r'\\bstoploss\\b', r'\\bstop-loss\\b', r'\\bstop\\s+loss\\b', r'\\bstop\\s*:\\s*\\d+', r'\\bsl\\s*:\\s*\\d+'
        ])
        tp_patterns = self.config.get('patterns', {}).get('tp', [
            r'\\btp\\b', r'\\btp1\\b', r'\\btp2\\b', r'\\btp3\\b', r'\\btp4\\b', r'\\btakeprofit\\b', r'\\btake-profit\\b', r'\\btake\\s+profit\\b', r'\\btarget\\b', r'\\btp\\s*:\\s*\\d+', r'\\btarget\\s*:\\s*\\d+'
        ])
        self.sl_regex = re.compile('|'.join(sl_patterns), re.IGNORECASE)
        self.tp_regex = re.compile('|'.join(tp_patterns), re.IGNORECASE)

    def has_sl_and_tp(self, text: str) -> Tuple[bool, bool]:
        """Check if text contains both SL and TP keywords using loaded patterns."""
        has_sl = bool(self.sl_regex.search(text))
        has_tp = bool(self.tp_regex.search(text))
        return has_sl, has_tp

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Gemini Signal Parser")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the configuration file (config.yaml)")
    args = parser.parse_args()
    
    signal_parser = None
    
    try:
        # Initialize SignalParser
        signal_parser = SignalParser(config_path=args.config)
        
        # Process each group
        for group_name in signal_parser.config['telegram']['groups']:
            signal_parser.process_group(group_name)
        
        # Print final statistics
        overall_stats = signal_parser.progress_tracker.get_overall_progress()
        signal_parser.logger.info(
            f"Processing completed - "
            f"Groups: {overall_stats['total_groups']}, "
            f"Completed: {overall_stats['completed_groups']}, "
            f"Total processed: {overall_stats['session_info']['total_processed']}, "
            f"Valid signals: {overall_stats['session_info']['total_valid_signals']}"
        )
        
        # Print rate limiter stats
        rate_limiter_stats = signal_parser.rate_limiter.get_stats()
        signal_parser.logger.info(
            f"Rate limiter statistics - "
            f"Total requests: {rate_limiter_stats['total_requests']}, "
            f"Successful: {rate_limiter_stats['successful_requests']}, "
            f"Failed: {rate_limiter_stats['failed_requests']}, "
            f"Retries: {rate_limiter_stats['retries']}"
        )
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        sys.exit(1)
    finally:
        if signal_parser and hasattr(signal_parser, 'rate_limiter'):
            signal_parser.rate_limiter.stop(wait=True)
            logging.info("Rate limiter stopped")

def get_default_group():
    """Get the first group from the groups directory"""
    groups_dir = Path("data/groups")
    if not groups_dir.exists():
        return "Sniper Fx signal group"
    
    for item in groups_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            return item.name
    
    return "Sniper Fx signal group"

if __name__ == "__main__":
    # Set defaults
    config_file = r"C:\Users\PC\OneDrive\Desktop\python\gemini_signal_extractor\config.yaml"
    # Parse arguments but make them all optional
    parser = argparse.ArgumentParser(description="Gemini Signal Parser")
    parser.add_argument("-c", "--config", default=config_file, help=f"Path to configuration file (default: {config_file})")
    parser.add_argument("-n", "--max-messages", type=int, help="Maximum number of messages to process (default: process all)")
    args = parser.parse_args()

    try:
        # Initialize parser
        signal_parser = SignalParser(args.config)

        # Loop over all groups in config
        for group_name in signal_parser.config['telegram']['groups']:
            signal_parser.process_group(
                group_name,
                max_messages=args.max_messages
            )

        # Stop rate limiter and save state
        signal_parser.rate_limiter.stop()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        if 'signal_parser' in locals():
            signal_parser.rate_limiter.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'signal_parser' in locals():
            signal_parser.rate_limiter.stop()
        sys.exit(1)