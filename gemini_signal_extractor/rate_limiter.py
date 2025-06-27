#!/usr/bin/env python3
"""
Gemini Rate Limiter
Handles rate limiting for Gemini API calls with robust error handling and progress tracking.
"""

import time
import json
import logging
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class GeminiRateLimiter:
    def __init__(self, config: Dict):
        """
        Initialize rate limiter with configuration
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config['rate_limiting']
        self.gemini_config = config['gemini']
        
        # --- Rate limiting config ---
        # Priority: gemini.max_requests_per_minute > rate_limiting.max_requests_per_minute > default 12
        self.max_requests_per_minute = (
            self.gemini_config.get('max_requests_per_minute')
            or self.config.get('max_requests_per_minute')
            or 12
        )
        self.config['max_requests_per_minute'] = self.max_requests_per_minute
        # Calculate request interval (seconds between requests)
        self.config['request_interval'] = 60.0 / self.max_requests_per_minute
        
        # Initialize queues
        self.request_queue = Queue()
        self.retry_queue = Queue()
        self.results = []
        
        # Control flags
        self.is_running = False
        self.worker_thread = None
        
        # State tracking
        self.last_request_time = time.time()
        self.request_count = 0
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries': 0,
            'start_time': None,
            'last_error': None
        }
        
        # Create state directory if needed
        state_dir = Path(config['paths']['data_dir']) / 'rate_limiter_state'
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = state_dir / 'state.json'
        
        # Load existing state if available
        self._load_state()
    
    def _load_state(self):
        """Load rate limiter state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.stats.update(state.get('stats', {}))
                    logger.info(f"Loaded rate limiter state from {self.state_file}")
        except Exception as e:
            logger.error(f"Error loading rate limiter state: {e}")
    
    def _save_state(self):
        """Save current state to file"""
        try:
            state = {
                'stats': self.stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving rate limiter state: {e}")
    
    def add_request(self, data: Any, callback: Optional[Callable] = None, process_fn: Optional[Callable] = None):
        """Add a request to the queue with optional callback and processor"""
        self.request_queue.put({
            'data': data,
            'callback': callback,
            'process_fn': process_fn,
            'attempts': 0,
            'added_at': datetime.now().isoformat()
        })
        logger.debug(f"Added request to queue. Queue size: {self.request_queue.qsize()}")
        
    def _process_request(self, request):
        """Process a request using its process function and callback"""
        try:
            if request['process_fn']:
                result = request['process_fn'](request['data'])
                if request['callback']:
                    return request['callback'](result)
            return None
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return None
            
    def _handle_request(self, request):
        """Handle a single request with rate limiting"""
        try:
            # Apply rate limiting
            self._check_rate_limit()
            
            # Process the request
            result = self._process_request(request)
            
            # Update stats
            self.stats['successful_requests'] += 1
            self.last_request_time = time.time()
            self.request_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.stats['failed_requests'] += 1
            if request['attempts'] < self.config.get('max_retries', 3):
                request['attempts'] += 1
                self.retry_queue.put(request)
            return None
    
    def _check_rate_limit(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Reset counter if a minute has passed
        if elapsed >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # If we've hit the limit, wait
        if self.request_count >= self.config['max_requests_per_minute']:
            sleep_time = 60 - elapsed
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
    
    def _worker_loop(self):
        """Main worker loop"""
        logger.info("Rate limiter worker started")
        self.stats['start_time'] = datetime.now().isoformat()
        
        while self.is_running:
            try:
                # Check retry queue first
                while not self.retry_queue.empty():
                    retry_request = self.retry_queue.get_nowait()
                    if datetime.now() >= datetime.fromisoformat(retry_request['retry_after']):
                        self._process_request(retry_request)
                        continue
                    self.retry_queue.put(retry_request)
                    break
                
                # Process main queue
                request = self.request_queue.get(timeout=1)
                self._process_request(request)
                
                # Rate limiting sleep
                request_interval = self.config.get('request_interval', 5.0)  # Default to 5 seconds if not set
                time.sleep(request_interval)
                
            except Empty:
                # No requests available
                if self.request_queue.empty() and self.retry_queue.empty():
                    time.sleep(1)
                continue
            
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the rate limiter worker"""
        if self.is_running:
            logger.warning("Rate limiter already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Rate limiter started")
    
    def stop(self, wait=True):
        """Stop the rate limiter worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        if wait and self.worker_thread:
            self.worker_thread.join(timeout=30)
        
        self._save_state()
        logger.info("Rate limiter stopped")
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = self.stats.copy()
        stats.update({
            'queue_size': self.request_queue.qsize(),
            'retry_queue_size': self.retry_queue.qsize(),
            'results_count': len(self.results)
        })
        return stats
