#!/usr/bin/env python3
"""
Trading Signal Parser
====================

A comprehensive tool to extract trading signals from CSV files containing chat conversations.
Uses Gemini AI for structured signal extraction with rate limiting and regex filtering.

Features:
- Regex filtering for SL (Stop Loss) and TP (Take Profit) keywords
- Rate-limited Gemini API integration (12 requests per minute)
- Index-by-index processing with comprehensive logging
- Dynamic path handling for multiple CSV files
- Individual signals CSV per window + master combined CSV
- Error handling and diagnostics

Usage:
    python signal_parser.py

Author: Trading Signal Extractor
Date: March 2025
"""

import os
import re
import time
import logging
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import deque
import json
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

class RateLimiter:
    """Rate limiter for Gemini API calls - 12 requests per minute"""
    
    def __init__(self, max_requests: int = 12, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than time_window
        while self.request_times and now - self.request_times[0] >= self.time_window:
            self.request_times.popleft()
            
        # If we're at the limit, wait until we can make another request
        if len(self.request_times) >= self.max_requests:
            sleep_time = self.time_window - (now - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                
        # Record this request
        self.request_times.append(now)

# Replace all self.logger with logger
logger = logging.getLogger(__name__)

class TradingSignalParser:
    """Main parser class for extracting trading signals from chat conversations"""
    
    def __init__(self, api_key: str):
        # Configure Gemini AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rate_limiter = RateLimiter()
        
        # Regex patterns for SL and TP keywords (case-insensitive)
        self.sl_patterns = [
            r'\bsl\b',
            r'\bstoploss\b',
            r'\bstop-loss\b',
            r'\bstop\s+loss\b',
            r'\bstop\s*:\s*\d+',
            r'\bsl\s*:\s*\d+'
        ]
        
        self.tp_patterns = [
            r'\btp\b',
            r'\btp1\b',
            r'\btp2\b',
            r'\btp3\b',
            r'\btp4\b',
            r'\btakeprofit\b',
            r'\btake-profit\b',
            r'\btake\s+profit\b',
            r'\btarget\b',
            r'\btp\s*:\s*\d+',
            r'\btarget\s*:\s*\d+'
        ]
        
        # Compile regex patterns
        self.sl_regex = re.compile('|'.join(self.sl_patterns), re.IGNORECASE)
        self.tp_regex = re.compile('|'.join(self.tp_patterns), re.IGNORECASE)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('signal_extraction.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)  # Use sys.stdout for UTF-8 support
            ]
        )
        logger = logging.getLogger(__name__)
    
    def has_sl_and_tp(self, text: str) -> Tuple[bool, bool]:
        """Check if text contains both SL and TP keywords"""
        has_sl = bool(self.sl_regex.search(text))
        has_tp = bool(self.tp_regex.search(text))
        return has_sl, has_tp
    
    def extract_signal_with_gemini(self, text: str) -> Dict[str, Any]:
        """Extract trading signal using Gemini AI"""
        prompt = f"""
        Extract trading signal information from this text: "{text}"
        
        Please return a JSON object with the following structure:
        {{
            "direction": "BUY" or "SELL" or null,
            "entry_price": number or null,
            "entry_zone": "price range" or null,
            "sl": number or null,
            "tp1": number or null,
            "tp2": number or null,
            "tp3": number or null,
            "tp4": number or null,
            "notes": "any corrections or observations",
            "confidence": "high/medium/low"
        }}
        
        Rules:
        1. If direction is not clear, set to null
        2. For entry, use entry_price if specific price, entry_zone if range
        3. Extract up to 4 take profit levels (tp1, tp2, tp3, tp4)
        4. Set unused TP levels to null
        5. Handle typos and variations in terminology
        6. Include any corrections made in notes
        7. Be flexible with ordering and format variations
        
        Return only valid JSON, no additional text.
        """
        
        try:
            self.rate_limiter.wait_if_needed()
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response if it has markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            parsed_response = json.loads(response_text)
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response.text}")
            return {
                "direction": None,
                "entry_price": None,
                "entry_zone": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "tp4": None,
                "notes": f"JSON parsing error: {str(e)}",
                "confidence": "low"
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "direction": None,
                "entry_price": None,
                "entry_zone": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "tp4": None,
                "notes": f"API error: {str(e)}",
                "confidence": "low"
            }
    
    def process_window_csv(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single window CSV file, saving after every batch of 12 requests and after sleep."""
        logger.info(f"Processing: {csv_path}")
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
        for idx in range(total_rows):
            try:
                text = str(df_window.loc[idx, 'combined_text'])
                has_sl, has_tp = self.has_sl_and_tp(text)
                if has_sl and has_tp:
                    logger.info(f"Processing index {idx} with Gemini AI")
                    parsed_signal = self.extract_signal_with_gemini(text)
                    signal_row = {
                        "window": window_name,
                        "original_index": idx,
                        "combined_text": text,
                        **parsed_signal,
                        "processed_at": datetime.now().isoformat()
                    }
                    signals_rows.append(signal_row)
                elif has_sl or has_tp:
                    reason = f"Has {'SL' if has_sl else 'TP'} but missing {'TP' if has_sl else 'SL'}"
                    skipped_rows.append({
                        "window": window_name,
                        "original_index": idx,
                        "combined_text": text,
                        "reason": reason,
                        "has_sl": has_sl,
                        "has_tp": has_tp
                    })
                else:
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
    
    def save_window_results(self, csv_path: str, df_signals: pd.DataFrame, 
                           df_skipped: pd.DataFrame, df_errors: pd.DataFrame):
        """Save results for a window in its directory"""
        window_dir = Path(csv_path).parent
        
        # Save signals
        if not df_signals.empty:
            signals_path = window_dir / "signals.csv"
            df_signals.to_csv(signals_path, index=False)
            logger.info(f"Saved signals to: {signals_path}")
        
        # Save skipped (for diagnostics)
        if not df_skipped.empty:
            skipped_path = window_dir / "skipped.csv"
            df_skipped.to_csv(skipped_path, index=False)
            logger.info(f"Saved skipped to: {skipped_path}")
        
        # Save errors (for debugging)
        if not df_errors.empty:
            errors_path = window_dir / "errors.csv"
            df_errors.to_csv(errors_path, index=False)
            logger.info(f"Saved errors to: {errors_path}")
    
    def process_all_windows(self, window_paths: List[str], master_out_dir: str):
        """Process all window CSV files and create master aggregation"""
        all_signals = []
        
        logger.info(f"Starting processing of {len(window_paths)} window files")
        
        for csv_path in window_paths:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {csv_path}")
            
            # Process this window
            df_signals, df_skipped, df_errors = self.process_window_csv(csv_path)
            
            # Save results in window directory
            self.save_window_results(csv_path, df_signals, df_skipped, df_errors)
            
            # Collect signals for master aggregation
            if not df_signals.empty:
                all_signals.append(df_signals)
        
        # Create master signals file
        if all_signals:
            df_master_signals = pd.concat(all_signals, ignore_index=True)
            
            # Save master signals
            master_path = Path(master_out_dir) / "final_signals.csv"
            df_master_signals.to_csv(master_path, index=False)
            
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING COMPLETE!")
            logger.info(f"Master signals saved to: {master_path}")
            logger.info(f"Total signals extracted: {len(df_master_signals)}")
            logger.info(f"Total windows processed: {len(window_paths)}")
            
            return df_master_signals
        else:
            logger.warning("No signals extracted from any window files")
            return pd.DataFrame()

def load_config(config_path: str):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """Main function to run the signal parser (fully config-driven)"""
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gemini_signal_extractor', 'config.yaml')
    config = load_config(config_path)

    # Paths and groups
    windows_template = config['paths']['windows_dir']
    groups = config['telegram']['groups']
    window_paths = []
    for group in groups:
        group_windows = windows_template.replace('{group_name}', group)
        group_windows_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gemini_signal_extractor', group_windows))
        if Path(group_windows_abs).exists():
            window_paths.append(group_windows_abs)
            print(f"✓ Found: {group_windows_abs}")
        else:
            print(f"✗ Not found: {group_windows_abs}")

    if not window_paths:
        print("Error: No valid windows.csv files found for any group.")
        return

    # Output directory
    master_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gemini_signal_extractor', os.path.dirname(config['paths']['final_signals_file'])))
    Path(master_out_dir).mkdir(parents=True, exist_ok=True)

    # API key
    api_key = os.getenv('GEMINI_API_KEY', config['gemini']['api_key'])
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables or config.yaml")
        return

    # Initialize parser (pass config-driven params if needed)
    parser = TradingSignalParser(api_key)

    # Process all windows
    try:
        df_master = parser.process_all_windows(window_paths, master_out_dir)
        if not df_master.empty:
            print(f"\n🎉 SUCCESS! Extracted {len(df_master)} trading signals")
            print(f"📁 Master file: {Path(master_out_dir) / 'final_signals.csv'}")
            print("\n📊 SUMMARY STATISTICS:")
            if 'direction' in df_master.columns:
                direction_counts = df_master['direction'].value_counts()
                print(f"   - BUY signals: {direction_counts.get('BUY', 0)}")
                print(f"   - SELL signals: {direction_counts.get('SELL', 0)}")
            if 'window' in df_master.columns:
                window_counts = df_master['window'].value_counts()
                print("   - Signals per window:")
                for window, count in window_counts.items():
                    print(f"     • {window}: {count}")
        else:
            print("⚠️  No signals extracted. Check the log files for details.")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()