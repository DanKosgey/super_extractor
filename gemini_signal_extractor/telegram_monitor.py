#!/usr/bin/env python3
"""
Live Telegram Signal Monitoring System

A real-time event-driven pipeline that monitors Telegram groups for trading signals,
processes them through time-based windows, and extracts structured signals using AI.

Features:
- Real-time message ingestion with zero data loss
- Time-based signal windowing (2-minute windows)
- AI-powered signal extraction using Gemini
- State persistence and crash recovery
- Rate limiting and async processing
- Comprehensive logging and monitoring
"""

import os
import json
import yaml
import asyncio
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
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

# Configuration and validation
VALID_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'GBPAUD', 'EURAUD', 'GBPCAD',
    'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD', 'US30', 'SPX500', 'NAS100',
    'GER30', 'UK100', 'JPN225', 'USOIL', 'UKOIL', 'NATGAS'
]

SIGNAL_KEYWORDS = [
    'buy', 'sell', 'long', 'short', 'entry', 'tp', 'take profit', 'sl', 
    'stop loss', 'breakeven', 'be', 'target', 'resistance', 'support'
]

@dataclass
class TradingSignal:
    """Trading signal data structure matching the existing schema"""
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
            'gemini.api_key', 'signal_extraction.time_window_minutes',
            'data_dir', 'logs_dir'
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
    """Represents a time-based signal window"""
    
    def __init__(self, window_id: str, trigger_message: Dict, group_name: str, 
                 window_duration: int = 2):
        self.window_id = window_id
        self.group_name = group_name
        self.start_time = datetime.fromisoformat(trigger_message['timestamp'])
        self.window_duration = timedelta(minutes=window_duration)
        self.end_time = self.start_time + self.window_duration
        self.messages = [trigger_message]
        self.is_complete = False
        self.is_expired = False
        self.parsed = False
    
    def add_message(self, message: Dict) -> bool:
        """Add message to window if within time bounds"""
        msg_time = datetime.fromisoformat(message['timestamp'])
        
        if msg_time <= self.end_time and not self.is_expired:
            self.messages.append(message)
            return True
        return False
    
    def check_completion(self) -> bool:
        """Check if window contains all required signal components"""
        if self.is_complete:
            return True
        
        combined_text = self.get_combined_text().lower()
        
        # Check for required components
        has_direction = any(keyword in combined_text for keyword in ['buy', 'sell', 'long', 'short'])
        has_sl = 'sl' in combined_text or 'stop loss' in combined_text
        has_tp = 'tp' in combined_text or 'take profit' in combined_text
        
        self.is_complete = has_direction and has_sl and has_tp
        return self.is_complete
    
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
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalWindow':
        """Restore window from dictionary"""
        window = cls.__new__(cls)
        window.window_id = data['window_id']
        window.group_name = data['group_name']
        window.start_time = datetime.fromisoformat(data['start_time'])
        window.end_time = datetime.fromisoformat(data['end_time'])
        window.messages = data['messages']
        window.is_complete = data['is_complete']
        window.is_expired = data['is_expired']
        window.parsed = data.get('parsed', False)
        window.window_duration = window.end_time - window.start_time
        return window

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
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Hello")
            logging.info("Gemini AI connection verified")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Gemini AI: {e}")
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build prompt for signal extraction"""
        return f"""
Extract trading signal information from the following text and return ONLY a valid JSON object.

Valid symbols: {', '.join(VALID_SYMBOLS)}

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
            
            model = genai.GenerativeModel('gemini-pro')
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
        if data['symbol'] not in VALID_SYMBOLS:
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
    
    def create_window(self, message: Dict, group_name: str) -> SignalWindow:
        """Create new signal window"""
        window_id = f"{group_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        window = SignalWindow(window_id, message, group_name, self.window_duration)
        
        with self._lock:
            self.active_windows[group_name].append(window)
        
        logging.info(f"Created new window {window_id} for group {group_name}")
        return window
    
    def add_message_to_windows(self, message: Dict, group_name: str):
        """Add message to all active windows for group"""
        with self._lock:
            windows_to_remove = []
            
            for window in self.active_windows[group_name]:
                if window.is_expired or window.is_complete:
                    continue
                
                if window.add_message(message):
                    logging.debug(f"Added message to window {window.window_id}")
                
                # Check if window is now complete or expired
                if window.check_completion() or window.check_expiry():
                    windows_to_remove.append(window)
            
            # Move completed/expired windows
            for window in windows_to_remove:
                self.active_windows[group_name].remove(window)
                self.completed_windows.append(window)
                
                if window.is_complete:
                    logging.info(f"Window {window.window_id} completed with {len(window.messages)} messages")
    
    def get_ready_windows(self) -> List[SignalWindow]:
        """Get windows ready for parsing"""
        ready = []
        
        with self._lock:
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
    
    def cleanup_old_windows(self, max_age_hours: int = 24):
        """Remove old windows to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            # Clean completed windows
            self.completed_windows = [
                w for w in self.completed_windows 
                if w.start_time > cutoff_time
            ]
            
            # Clean active windows (shouldn't happen but safety check)
            for group_name in self.active_windows:
                self.active_windows[group_name] = [
                    w for w in self.active_windows[group_name]
                    if w.start_time > cutoff_time
                ]
    
    def get_state(self) -> Dict:
        """Get current state for persistence"""
        with self._lock:
            return {
                'active_windows': {
                    group: [w.to_dict() for w in windows]
                    for group, windows in self.active_windows.items()
                },
                'completed_windows': [w.to_dict() for w in self.completed_windows]
            }
    
    def restore_state(self, state: Dict):
        """Restore state from persistence"""
        with self._lock:
            # Restore active windows
            for group_name, windows_data in state.get('active_windows', {}).items():
                self.active_windows[group_name] = [
                    SignalWindow.from_dict(w_data) for w_data in windows_data
                ]
            
            # Restore completed windows
            self.completed_windows = [
                SignalWindow.from_dict(w_data) 
                for w_data in state.get('completed_windows', [])
            ]

class TelegramListener:
    """Real-time Telegram message listener"""
    
    def __init__(self, config: Dict, window_manager: WindowManager, 
                 state_manager: StateManager):
        self.config = config
        self.window_manager = window_manager
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        api_id = config['telegram']['api_id']
        api_hash = config['telegram']['api_hash']
        self.client = TelegramClient('live_session', api_id, api_hash)
        
        self.group_ids = config['telegram']['group_ids']
        self.active_groups = config['telegram']['groups']
        self.last_message_ids = {}
        
        # Data directories
        self.data_dir = Path(config['data_dir'])
        self.groups_dir = self.data_dir / 'groups'
        self.groups_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_signal_trigger(self, text: str) -> bool:
        """Check if message contains signal trigger keywords or symbols"""
        if not text:
            return False
        
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Check for signal keywords
        for keyword in SIGNAL_KEYWORDS:
            if keyword in text_lower:
                return True
        
        # Check for valid symbols
        for symbol in VALID_SYMBOLS:
            if symbol in text_upper:
                return True
        
        return False
    
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
            
            # Check if this is a signal trigger
            if self._is_signal_trigger(message.text):
                self.window_manager.create_window(message_data, group_name)
                self.logger.info(f"Signal trigger detected in {group_name}: {message.text[:100]}...")
            
            # Add to existing windows
            self.window_manager.add_message_to_windows(message_data, group_name)
            
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
                entity = await self.client.get_entity(chat_id)
                chat_entities.append(entity)
        
        # Register event handler
        @self.client.on(events.NewMessage(chats=chat_entities, incoming=True))
        async def message_handler(event):
            await self._handle_new_message(event)
        
        self.logger.info(f"Started listening to {len(chat_entities)} groups")
        
        # Keep running
        await self.client.run_until_disconnected()
    
    async def catch_up_history(self):
        """Catch up on missed messages since last run"""
        state = self.state_manager.load_state()
        
        for group_name in self.active_groups:
            if group_name not in self.group_ids:
                continue
            
            last_id = state.get('groups', {}).get(group_name, {}).get('last_message_id', 0)
            chat_id = self.group_ids[group_name]
            
            try:
                entity = await self.client.get_entity(chat_id)
                
                # Get latest messages
                messages = await self.client.get_messages(entity, limit=100)
                
                new_messages = []
                for msg in reversed(messages):
                    if msg.id > last_id and msg.text:
                        new_messages.append(msg)
                
                self.logger.info(f"Catching up {len(new_messages)} messages for {group_name}")
                
                # Process new messages
                for msg in new_messages:
                    message_data = {
                        'message_id': msg.id,
                        'timestamp': msg.date.isoformat(),
                        'text': msg.text,
                        'group_name': group_name
                    }
                    
                    self._save_raw_message(message_data, group_name)
                    
                    if self._is_signal_trigger(msg.text):
                        self.window_manager.create_window(message_data, group_name)
                    
                    self.window_manager.add_message_to_windows(message_data, group_name)
                    
                    self.last_message_ids[group_name] = msg.id
                
            except Exception as e:
                self.logger.error(f"Error catching up history for {group_name}: {e}")

class SignalOutput:
    """Handles signal output and CSV management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.signals_dir = self.data_dir / 'final_signals'
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        self.final_signals_file = self.signals_dir / 'all_signals.csv'
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
            # Save to final signals file
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
            group_signals_file = group_dir / 'signals.csv'
            
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
        self.state_manager = State