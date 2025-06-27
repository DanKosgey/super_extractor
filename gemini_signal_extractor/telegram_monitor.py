#!/usr/bin/env python3
"""
Enhanced Telegram Signal Monitoring System

A real-time monitoring system that:
1. Monitors Telegram groups for new messages
2. Detects direction keywords (buy, sell, long, short)
3. If message has direction + SL + TP -> sends immediately to AI for parsing
4. If message has only direction -> opens 2-minute window to collect additional messages
5. Within window, checks for SL/TP completion and sends to AI when complete
6. Discards incomplete signals after window expires
7. Saves all signals to CSV files

Perfect accuracy in window handling to avoid missing signals.
"""

import os
import json
import yaml
import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import time
import re
import csv
import uuid

# Telethon imports
from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel

# AI imports
import google.generativeai as genai

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    group_name: str
    window_id: str
    timestamp: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    confidence: float = 0.0
    valid: bool = True
    corrections: str = ""
    validation_notes: str = ""
    context: str = ""
    raw_text: str = ""

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def _validate_config(self):
        """Validate required configuration keys"""
        required_keys = [
            'telegram.api_id', 'telegram.api_hash', 'telegram.phone',
            'telegram.group_ids', 'telegram.groups',
            'gemini.api_key', 'signal_extraction.time_window_minutes'
        ]
        
        for key_path in required_keys:
            keys = key_path.split('.')
            current = self.config
            
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config key: {key_path}")
                current = current[key]

class StateManager:
    """Manages persistent state for crash recovery"""
    
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "monitor_state.json"
        self._lock = threading.Lock()
    
    def load_state(self) -> Dict:
        """Load state from disk"""
        with self._lock:
            if not self.state_file.exists():
                return {"groups": {}, "windows": {}}
            
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load state: {e}")
                return {"groups": {}, "windows": {}}
    
    def save_state(self, state: Dict):
        """Save state to disk atomically"""
        with self._lock:
            temp_file = self.state_file.with_suffix('.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, default=str)
                temp_file.replace(self.state_file)
            except Exception as e:
                logging.error(f"Failed to save state: {e}")
                if temp_file.exists():
                    temp_file.unlink()

class SignalWindow:
    """Represents a time-based signal window for collecting messages"""
    
    def __init__(self, window_id: str, trigger_message: Dict, group_name: str, 
                 window_duration: int = 2):
        self.window_id = window_id
        self.group_name = group_name
        self.trigger_message = trigger_message
        self.start_time = datetime.fromisoformat(trigger_message['timestamp'])
        self.window_duration = timedelta(minutes=window_duration)
        self.end_time = self.start_time + self.window_duration
        self.messages = [trigger_message]
        self.is_complete = False
        self.is_expired = False
        self.parsed = False
        
        # Check if trigger message is already complete
        self.is_complete = self._check_completion()
    
    def add_message(self, message: Dict) -> bool:
        """Add message to window if within time bounds"""
        if self.is_expired or self.is_complete:
            return False
            
        msg_time = datetime.fromisoformat(message['timestamp'])
        
        if msg_time <= self.end_time:
            self.messages.append(message)
            # Check if window is now complete
            if self._check_completion():
                self.is_complete = True
                logging.info(f"Window {self.window_id} completed with {len(self.messages)} messages")
            return True
        return False
    
    def _check_completion(self) -> bool:
        """Check if window contains all required signal components"""
        combined_text = self.get_combined_text().lower()
        
        # Check for direction keywords
        direction_keywords = ['buy', 'sell', 'long', 'short']
        has_direction = any(keyword in combined_text for keyword in direction_keywords)
        
        # Check for SL keywords
        sl_keywords = ['sl', 'stop loss', 'stop-loss', 'stoploss']
        has_sl = any(keyword in combined_text for keyword in sl_keywords)
        
        # Check for TP keywords (including tp1, tp2, etc.)
        tp_keywords = ['tp', 'tp1', 'tp2', 'tp3', 'tp4', 'take profit', 'takeprofit', 'target']
        has_tp = any(keyword in combined_text for keyword in tp_keywords)
        
        return has_direction and has_sl and has_tp
    
    def check_expiry(self) -> bool:
        """Check if window has expired"""
        if not self.is_expired:
            self.is_expired = datetime.now() > self.end_time
        return self.is_expired
    
    def get_combined_text(self) -> str:
        """Get all messages combined into one text block"""
        return ' '.join(msg['text'] for msg in self.messages if msg['text'])
    
    def to_dict(self) -> Dict:
        """Convert window to dictionary for serialization"""
        return {
            'window_id': self.window_id,
            'group_name': self.group_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'messages': self.messages,
            'is_complete': self.is_complete,
            'is_expired': self.is_expired,
            'parsed': self.parsed
        }

class GeminiRateLimiter:
    """Rate limiter for Gemini API calls"""
    
    def __init__(self, max_requests_per_minute: int = 15):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self._lock = threading.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self._lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request) + 1
                
        if len(self.requests) >= self.max_requests:
            logging.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        with self._lock:
            self.requests.append(time.time())

class SignalParser:
    """AI-powered signal parser using Gemini"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rate_limiter = GeminiRateLimiter(
            config.get('gemini', {}).get('max_requests_per_minute', 15)
        )
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize Gemini AI"""
        api_key = self.config['gemini']['api_key']
        genai.configure(api_key=api_key)
        
        # Test connection
        try:
            model_name = self.config['gemini'].get('model', 'gemini-1.5-flash')
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello")
            logging.info("Gemini AI connection verified")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini AI: {e}")
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for signal extraction"""
        valid_symbols = ', '.join(self.config['signal_extraction']['valid_symbols'])
        
        return f"""
Extract trading signal information from the following text and return ONLY a valid JSON object.

Valid symbols: {valid_symbols}

Required JSON format:
{{
    "symbol": "SYMBOL_NAME",
    "direction": "BUY|SELL|LONG|SHORT",
    "entry": 1.2345,
    "sl": 1.2300,
    "tp1": 1.2400,
    "tp2": 1.2450,
    "tp3": 1.2500,
    "tp4": null,
    "confidence": 0.85
}}

Rules:
1. Only extract if ALL required fields (symbol, direction, entry, sl, tp1) are present
2. For BUY/LONG: sl < entry < tp1 < tp2 < tp3 < tp4
3. For SELL/SHORT: sl > entry > tp1 > tp2 > tp3 > tp4
4. Set confidence 0.0-1.0 based on signal clarity
5. Use null for missing TP levels
6. Return only the JSON object, no other text

Text to analyze:
{text}
"""
    
    async def parse_signal(self, window: SignalWindow) -> Optional[TradingSignal]:
        """Parse signal from window using Gemini AI"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            combined_text = window.get_combined_text()
            prompt = self._build_extraction_prompt(combined_text)
            
            model_name = self.config['gemini'].get('model', 'gemini-1.5-flash')
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            if not response.text:
                logging.warning(f"Empty response from Gemini for window {window.window_id}")
                return None
            
            # Parse JSON response
            try:
                signal_data = json.loads(response.text.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    signal_data = json.loads(json_match.group())
                else:
                    logging.error(f"Invalid JSON response for window {window.window_id}")
                    return None
            
            # Validate and create TradingSignal
            if self._validate_signal_data(signal_data):
                return TradingSignal(
                    group_name=window.group_name,
                    window_id=window.window_id,
                    timestamp=window.start_time.isoformat(),
                    symbol=signal_data['symbol'],
                    direction=signal_data['direction'].upper(),
                    entry=float(signal_data['entry']),
                    sl=float(signal_data['sl']),
                    tp1=float(signal_data['tp1']),
                    tp2=float(signal_data['tp2']) if signal_data.get('tp2') else None,
                    tp3=float(signal_data['tp3']) if signal_data.get('tp3') else None,
                    tp4=float(signal_data['tp4']) if signal_data.get('tp4') else None,
                    confidence=float(signal_data.get('confidence', 0.0)),
                    context=f"Window: {window.window_id}, Messages: {len(window.messages)}",
                    raw_text=combined_text
                )
            
            return None
            
        except Exception as e:
            logging.error(f"Error parsing signal for window {window.window_id}: {e}")
            return None
    
    def _validate_signal_data(self, data: Dict) -> bool:
        """Validate extracted signal data"""
        required_fields = ['symbol', 'direction', 'entry', 'sl', 'tp1']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Validate symbol
        valid_symbols = self.config['signal_extraction']['valid_symbols']
        if data['symbol'] not in valid_symbols:
            return False
        
        # Validate direction
        if data['direction'].upper() not in ['BUY', 'SELL', 'LONG', 'SHORT']:
            return False
        
        # Validate price relationships
        try:
            entry = float(data['entry'])
            sl = float(data['sl'])
            tp1 = float(data['tp1'])
            
            if data['direction'].upper() in ['BUY', 'LONG']:
                if not (sl < entry < tp1):
                    return False
            else:  # SELL/SHORT
                if not (sl > entry > tp1):
                    return False
        except (ValueError, TypeError):
            return False
        
        return True

class WindowManager:
    """Manages signal windows and their lifecycle"""
    
    def __init__(self, window_duration: int = 2):
        self.window_duration = window_duration
        self.active_windows: Dict[str, List[SignalWindow]] = defaultdict(list)
        self.completed_windows: List[SignalWindow] = []
        self._lock = threading.Lock()
        
        # Start background cleanup task
        self._cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_task.start()
    
    def _has_direction_keyword(self, text: str) -> bool:
        """Check if text contains direction keywords"""
        text_lower = text.lower()
        direction_keywords = ['buy', 'sell', 'long', 'short']
        return any(keyword in text_lower for keyword in direction_keywords)
    
    def _has_sl_keyword(self, text: str) -> bool:
        """Check if text contains SL keywords"""
        text_lower = text.lower()
        sl_keywords = ['sl', 'stop loss', 'stop-loss', 'stoploss']
        return any(keyword in text_lower for keyword in sl_keywords)
    
    def _has_tp_keyword(self, text: str) -> bool:
        """Check if text contains TP keywords"""
        text_lower = text.lower()
        tp_keywords = ['tp', 'tp1', 'tp2', 'tp3', 'tp4', 'take profit', 'takeprofit', 'target']
        return any(keyword in text_lower for keyword in tp_keywords)
    
    def _is_complete_signal(self, text: str) -> bool:
        """Check if text contains all required signal components"""
        return (self._has_direction_keyword(text) and 
                self._has_sl_keyword(text) and 
                self._has_tp_keyword(text))
    
    def process_message(self, message: Dict, group_name: str) -> Optional[SignalWindow]:
        """Process a new message and return completed window if any"""
        text = message.get('text', '').strip()
        if not text:
            return None
        
        with self._lock:
            # First, try to add message to existing active windows
            for window in self.active_windows[group_name][:]:  # Copy list to avoid modification during iteration
                if not window.is_complete and not window.is_expired:
                    if window.add_message(message):
                        if window.is_complete:
                            # Window completed, move to completed list
                            self.active_windows[group_name].remove(window)
                            self.completed_windows.append(window)
                            logging.info(f"Window {window.window_id} completed with message: {text[:100]}...")
                            return window
            
            # Check if this message should trigger a new window
            if self._has_direction_keyword(text):
                # Check if it's already a complete signal
                if self._is_complete_signal(text):
                    # Complete signal - create and immediately complete window
                    window_id = f"{group_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    window = SignalWindow(window_id, message, group_name, self.window_duration)
                    window.is_complete = True
                    self.completed_windows.append(window)
                    logging.info(f"Complete signal detected, created window {window_id}: {text[:100]}...")
                    return window
                else:
                    # Direction only - create new window for waiting
                    window_id = f"{group_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    window = SignalWindow(window_id, message, group_name, self.window_duration)
                    self.active_windows[group_name].append(window)
                    logging.info(f"Direction keyword detected, created waiting window {window_id}: {text[:100]}...")
        
        return None
    
    def _cleanup_loop(self):
        """Background cleanup of expired windows"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                self._cleanup_expired_windows()
            except Exception as e:
                logging.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_expired_windows(self):
        """Remove expired windows that never completed"""
        with self._lock:
            for group_name in list(self.active_windows.keys()):
                expired_windows = []
                for window in self.active_windows[group_name][:]:
                    if window.check_expiry():
                        expired_windows.append(window)
                        self.active_windows[group_name].remove(window)
                
                if expired_windows:
                    logging.info(f"Cleaned up {len(expired_windows)} expired windows for {group_name}")
    
    def get_ready_windows(self) -> List[SignalWindow]:
        """Get windows ready for parsing"""
        with self._lock:
            ready = []
            for window in self.completed_windows[:]:
                if window.is_complete and not window.parsed:
                    ready.append(window)
            return ready
    
    def mark_window_parsed(self, window_id: str):
        """Mark window as parsed"""
        with self._lock:
            for window in self.completed_windows:
                if window.window_id == window_id:
                    window.parsed = True
                    break

class SignalOutput:
    """Handles signal output and CSV management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.signals_dir = self.data_dir / 'signals'
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        self.final_signals_file = self.signals_dir / 'live_signals.csv'
        self._ensure_signals_file_exists()
    
    def _ensure_signals_file_exists(self):
        """Create signals CSV file with headers if it doesn't exist"""
        if not self.final_signals_file.exists():
            headers = [
                'group_name', 'window_id', 'timestamp', 'symbol', 'direction',
                'entry', 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'confidence',
                'valid', 'corrections', 'validation_notes', 'context', 'raw_text'
            ]
            
            with open(self.final_signals_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def save_signal(self, signal: TradingSignal):
        """Save signal to CSV files"""
        try:
            # Save to main signals file
            with open(self.final_signals_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                signal_dict = asdict(signal)
                writer.writerow([signal_dict.get(field, '') for field in [
                    'group_name', 'window_id', 'timestamp', 'symbol', 'direction',
                    'entry', 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'confidence',
                    'valid', 'corrections', 'validation_notes', 'context', 'raw_text'
                ]])
            
            # Save to group-specific file
            group_dir = self.data_dir / 'groups' / signal.group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            group_signals_file = group_dir / 'live_signals.csv'
            
            if not group_signals_file.exists():
                with open(group_signals_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'group_name', 'window_id', 'timestamp', 'symbol', 'direction',
                        'entry', 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'confidence',
                        'valid', 'corrections', 'validation_notes', 'context', 'raw_text'
                    ])
            
            with open(group_signals_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                signal_dict = asdict(signal)
                writer.writerow([signal_dict.get(field, '') for field in [
                    'group_name', 'window_id', 'timestamp', 'symbol', 'direction',
                    'entry', 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'confidence',
                    'valid', 'corrections', 'validation_notes', 'context', 'raw_text'
                ]])
            
            logging.info(f"Saved signal: {signal.symbol} {signal.direction} from {signal.group_name}")
            
        except Exception as e:
            logging.error(f"Error saving signal: {e}")

class TelegramListener:
    """Real-time Telegram message listener"""
    
    def __init__(self, config: Dict, window_manager: WindowManager, 
                 signal_parser: SignalParser, signal_output: SignalOutput,
                 state_manager: StateManager):
        self.config = config
        self.window_manager = window_manager
        self.signal_parser = signal_parser
        self.signal_output = signal_output
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        api_id = config['telegram']['api_id']
        api_hash = config['telegram']['api_hash']
        self.client = TelegramClient('signal_monitor_session', api_id, api_hash)
        
        self.group_ids = config['telegram']['group_ids']
        self.active_groups = config['telegram']['groups']
        self.last_message_ids = {}
        
        # Data directories
        self.data_dir = Path(config['paths']['data_dir'])
        self.groups_dir = self.data_dir / 'groups'
        self.groups_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_raw_message(self, message_data: Dict, group_name: str):
        """Save message to raw CSV file"""
        group_dir = self.groups_dir / group_name
        group_dir.mkdir(exist_ok=True)
        raw_csv_path = group_dir / 'raw_messages.csv'
        
        # Create CSV if it doesn't exist
        if not raw_csv_path.exists():
            with open(raw_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['message_id', 'timestamp', 'message'])
        
        # Append message
        with open(raw_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                message_data['message_id'],
                message_data['timestamp'],
                message_data['text']
            ])
    
    async def _handle_new_message(self, event):
        """Handle incoming Telegram message"""
        try:
            message = event.message
            if not message.text:
                return
            
            # Get group name
            chat = await event.get_chat()
            group_name = None
            
            for name, group_id in self.group_ids.items():
                if chat.id == group_id:
                    group_name = name
                    break
            
            if not group_name or group_name not in self.active_groups:
                return
            
            # Create message data
            message_data = {
                'message_id': message.id,
                'timestamp': message.date.isoformat(),
                'text': message.text,
                'group_name': group_name
            }
            
            # Save to raw CSV immediately
            self._save_raw_message(message_data, group_name)
            
            # Update last message ID
            self.last_message_ids[group_name] = message.id
            
            # Process message through window manager
            completed_window = self.window_manager.process_message(message_data, group_name)
            
            # If window completed, send to AI for parsing
            if completed_window:
                try:
                    signal = await self.signal_parser.parse_signal(completed_window)
                    if signal:
                        self.signal_output.save_signal(signal)
                        self.logger.info(f"Parsed and saved signal from window {completed_window.window_id}")
                    else:
                        self.logger.warning(f"Failed to parse signal from window {completed_window.window_id}")
                    
                    # Mark window as parsed
                    self.window_manager.mark_window_parsed(completed_window.window_id)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing signal from window {completed_window.window_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def connect(self):
        """Connect to Telegram"""
        await self.client.connect()
        
        if not await self.client.is_user_authorized():
            phone = self.config['telegram']['phone']
            await self.client.send_code_request(phone)
            code = input('Enter the code you received: ')
            await self.client.sign_in(phone, code)
        
        self.logger.info("Connected to Telegram")
    
    async def start_listening(self):
        """Start real-time message listening"""
        # Get chat entities
        chat_entities = []
        for group_name in self.active_groups:
            if group_name in self.group_ids:
                chat_id = self.group_ids[group_name]
                try:
                    entity = await self.client.get_entity(chat_id)
                    chat_entities.append(entity)
                    self.logger.info(f"Added group {group_name} to monitoring")
                except Exception as e:
                    self.logger.error(f"Failed to get entity for group {group_name}: {e}")
        
        if not chat_entities:
            self.logger.error("No chat entities found. Cannot start monitoring.")
            return
        
        # Register event handler
        @self.client.on(events.NewMessage(chats=chat_entities, incoming=True))
        async def message_handler(event):
            await self._handle_new_message(event)
        
        self.logger.info(f"Started listening to {len(chat_entities)} groups: {self.active_groups}")
        
        # Start background processing of any missed completed windows
        asyncio.create_task(self._background_processor())
        
        # Keep running
        await self.client.run_until_disconnected()
    
    async def _background_processor(self):
        """Process any completed windows that haven't been parsed yet"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                ready_windows = self.window_manager.get_ready_windows()
                for window in ready_windows:
                    try:
                        signal = await self.signal_parser.parse_signal(window)
                        if signal:
                            self.signal_output.save_signal(signal)
                            self.logger.info(f"Background processed signal from window {window.window_id}")
                        
                        self.window_manager.mark_window_parsed(window.window_id)
                        
                    except Exception as e:
                        self.logger.error(f"Error in background processing window {window.window_id}: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error in background processor: {e}")

class LiveTelegramMonitor:
    """Main monitoring system coordinator"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.state_manager = StateManager(Path(self.config['paths']['data_dir']))
        self.window_manager = WindowManager(
            self.config['signal_extraction']['time_window_minutes']
        )
        self.signal_parser = SignalParser(self.config)
        self.signal_output = SignalOutput(self.config)
        self.telegram_listener = TelegramListener(
            self.config, 
            self.window_manager, 
            self.signal_parser, 
            self.signal_output,
            self.state_manager
        )
        
        self.logger.info("LiveTelegramMonitor initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging_config = self.config.get('logging', {})
        
        # Create logs directory
        logs_dir = Path(self.config['paths']['logs_dir'])
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = logs_dir / logging_config.get('file', 'telegram_monitor.log')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Set third-party loggers to WARNING to reduce noise
        logging.getLogger('telethon').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    async def start(self):
        """Start the monitoring system"""
        try:
            self.logger.info("Starting Telegram Signal Monitor...")
            
            # Connect to Telegram
            await self.telegram_listener.connect()
            
            # Start listening
            await self.telegram_listener.start_listening()
            
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in monitoring system: {e}")
            raise
    
    def run(self):
        """Run the monitoring system"""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            raise

async def main():
    """Main entry point"""
    monitor = LiveTelegramMonitor()
    await monitor.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.error(f"Fatal error: {e}")