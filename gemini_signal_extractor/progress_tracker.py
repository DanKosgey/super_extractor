#!/usr/bin/env python3
"""
Progress Tracker Module
Manages and persists progress during the parsing process to ensure resumable operations.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import time
import shutil


class ProgressTracker:
    """
    Utility class to manage and persist progress during the parsing process.
    Ensures that the parser can resume from the last saved state in case of interruptions.
    """
    
    def __init__(self, progress_file: str):
        """
        Initialize the ProgressTracker with a file path where progress data is stored.
        
        Args:
            progress_file (str): Path to the progress file (e.g., JSON file)
        """
        self.progress_file = Path(progress_file)
        self.logger = logging.getLogger(__name__)
        self.state = {
            "groups": {},
            "processed_windows": {},
            "completed_groups": [],
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "total_processed": 0,
                "total_valid_signals": 0
            }
        }
        
        # Ensure the directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load existing state
        self._load_state()
    
    def _load_state(self):

        """Load state from JSON file or initialize new state"""
        try:

            if self.progress_file.exists():

                with open(self.progress_file, 'r', encoding='utf-8') as f:

                    loaded_state = json.load(f)
                    for key in self.state:
                        if key in loaded_state:
                            self.state[key] = loaded_state[key]
                    self.logger.info(f"Loaded existing progress from {self.progress_file}")
                
            else:
                self._save_state()
                self.logger.info(f"Initialized new progress file at {self.progress_file}")
        except Exception as e:
            self.logger.error(f"Error loading progress file: {e}")
            self._save_state()
    
    def _save_state(self):
        """Save current state to JSON file with atomic write, with fallback if atomic replace fails."""
        try:
            # Update timestamp
            self.state["last_updated"] = datetime.now().isoformat()
            temp_file = self.progress_file.with_suffix('.tmp')
            # Write to temporary file first
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            try:
                # Try atomic replace
                temp_file.replace(self.progress_file)
            except Exception as e:
                self.logger.warning(f"Atomic replace failed: {e}. Trying copy+remove fallback.")
                try:
                    # Fallback: copy then remove temp
                    import shutil
                    shutil.copyfile(temp_file, self.progress_file)
                    temp_file.unlink(missing_ok=True)
                except Exception as e2:
                    self.logger.error(f"Fallback save also failed: {e2}")
            self.logger.debug("Progress state saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving progress state: {e}")
    
    def are_all_windows_processed(self, group_name: str, all_window_ids: list) -> bool:
        """Return True if all window_ids for a group are marked as processed."""
        processed = set(self.state["processed_windows"].get(group_name, []))
        return set(map(str, all_window_ids)) <= processed

    def is_group_completed(self, group_name: str, all_window_ids: list = None) -> bool:
        """Check if a group has been fully processed (optionally, require all windows processed)."""
        if all_window_ids is not None:
            return self.are_all_windows_processed(group_name, all_window_ids)
        return group_name in self.state["completed_groups"]
    
    def is_window_processed(self, group_name: str, window_id: str) -> bool:
        """Check if a specific window has been processed"""
        return (group_name in self.state["processed_windows"] and
                window_id in self.state["processed_windows"][group_name])
        
    def mark_completed(self, group_name: str, window_id: str):
        """Mark a window as processed and log the update for diagnostics."""
        window_id = str(window_id)  # Always use string IDs for consistency
        if group_name not in self.state["processed_windows"]:
            self.state["processed_windows"][group_name] = []
        if window_id not in self.state["processed_windows"][group_name]:
            self.state["processed_windows"][group_name].append(window_id)
            self.state["session_info"]["total_processed"] += 1
            self._save_state()
            self.logger.debug(f"Marked window_id={window_id} as processed for group={group_name}. Now processed: {self.state['processed_windows'][group_name]}")
        else:
            self.logger.debug(f"Window_id={window_id} for group={group_name} was already marked as processed.")
    
    def mark_group_completed(self, group_name: str):
        """Mark an entire group as completed"""
        if group_name not in self.state["completed_groups"]:
            self.state["completed_groups"].append(group_name)
            self._save_state()
    
    def get_group_stats(self, group_name: str) -> Dict:
        """Get statistics for a specific group"""
        return self.state["groups"].get(group_name, {
            "total_signals": 0,
            "valid_signals": 0,
            "processed_windows": 0
        })
    
    def get_overall_progress(self) -> Dict:
        """Get overall processing statistics"""
        return {
            "total_groups": len(self.state["groups"]),
            "completed_groups": len(self.state["completed_groups"]),
            "session_info": self.state["session_info"],
            "last_updated": self.state.get("last_updated")
        }
    
    def clear_progress(self, group_name: Optional[str] = None):
        """
        Clear progress data
        
        Args:
            group_name (str, optional): If provided, only clear this group's data
        """
        if group_name:
            if group_name in self.state["groups"]:
                del self.state["groups"][group_name]
            if group_name in self.state["processed_windows"]:
                del self.state["processed_windows"][group_name]
            if group_name in self.state["completed_groups"]:
                self.state["completed_groups"].remove(group_name)
        else:
            # Reset to initial state
            self.state = {
                "groups": {},
                "processed_windows": {},
                "completed_groups": [],
                "session_info": {
                    "start_time": datetime.now().isoformat(),
                    "total_processed": 0,
                    "total_valid_signals": 0
                }
            }
        
        self._save_state()
    
    def update_group_progress(self, group_name: str, window_id: str, signal_count: int = 0, valid_count: int = 0):
        """Update progress for a group and log the update for diagnostics."""
        window_id = str(window_id)  # Always use string IDs for consistency
        if group_name not in self.state["groups"]:
            self.state["groups"][group_name] = {
                "processed_windows": [],
                "total_signals": 0,
                "valid_signals": 0,
                "start_time": datetime.now().isoformat()
            }
        if window_id not in self.state["groups"][group_name]["processed_windows"]:
            self.state["groups"][group_name]["processed_windows"].append(window_id)
            self.state["groups"][group_name]["total_signals"] += signal_count
            self.state["groups"][group_name]["valid_signals"] += valid_count
            self._save_state()
            self.logger.debug(f"Updated group progress for group={group_name}, window_id={window_id}. Now processed: {self.state['groups'][group_name]['processed_windows']}")
        else:
            self.logger.debug(f"Group progress for group={group_name}, window_id={window_id} was already updated.")
    
    def get_resume_index(self, group_name: str, all_window_ids: list) -> int:
        """Return the index of the first unprocessed window for a group."""
        processed = set(self.state["processed_windows"].get(group_name, []))
        for idx, window_id in enumerate(map(str, all_window_ids)):
            if window_id not in processed:
                return idx
        return len(all_window_ids)  # All processed

    def get_last_processed_index(self, group_name: str, all_window_ids: list) -> int:
        """Return the index of the last processed window for a group, or -1 if none."""
        processed = set(self.state["processed_windows"].get(group_name, []))
        last_idx = -1
        for idx, window_id in enumerate(map(str, all_window_ids)):
            if window_id in processed:
                last_idx = idx
        return last_idx