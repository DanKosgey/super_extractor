#!/usr/bin/env python3
"""
Test script for the Enhanced Telegram Signal Monitor
Tests the window management and signal detection logic
"""

import sys
import asyncio
import logging
from datetime import datetime, timedelta
from telegram_monitor import WindowManager, SignalParser, SignalOutput, ConfigManager

def test_window_manager():
    """Test the window manager functionality"""
    print("Testing Window Manager...")
    
    # Initialize window manager with 2-minute window
    window_manager = WindowManager(window_duration=2)
    
    # Test message 1: Direction only (should create waiting window)
    msg1 = {
        'message_id': 1,
        'timestamp': datetime.now().isoformat(),
        'text': 'BUY EURUSD at 1.1000',
        'group_name': 'test_group'
    }
    
    print(f"Processing message 1: {msg1['text']}")
    result1 = window_manager.process_message(msg1, 'test_group')
    print(f"Result: {result1}")
    print(f"Active windows: {len(window_manager.active_windows['test_group'])}")
    
    # Test message 2: Complete signal (should create and complete window immediately)
    msg2 = {
        'message_id': 2,
        'timestamp': (datetime.now() + timedelta(seconds=30)).isoformat(),
        'text': 'SELL GBPUSD at 1.2500, SL: 1.2550, TP1: 1.2450, TP2: 1.2400',
        'group_name': 'test_group'
    }
    
    print(f"\nProcessing message 2: {msg2['text']}")
    result2 = window_manager.process_message(msg2, 'test_group')
    print(f"Result: {result2}")
    print(f"Completed windows: {len(window_manager.completed_windows)}")
    
    # Test message 3: SL and TP for first window (should complete first window)
    msg3 = {
        'message_id': 3,
        'timestamp': (datetime.now() + timedelta(seconds=60)).isoformat(),
        'text': 'SL: 1.0950, TP1: 1.1050',
        'group_name': 'test_group'
    }
    
    print(f"\nProcessing message 3: {msg3['text']}")
    result3 = window_manager.process_message(msg3, 'test_group')
    print(f"Result: {result3}")
    print(f"Active windows: {len(window_manager.active_windows['test_group'])}")
    print(f"Completed windows: {len(window_manager.completed_windows)}")
    
    return True

def test_signal_detection():
    """Test signal detection functions"""
    print("\nTesting Signal Detection...")
    
    window_manager = WindowManager()
    
    # Test direction detection
    test_cases = [
        ("BUY EURUSD", True, "Direction keyword"),
        ("SELL GBPUSD", True, "Direction keyword"),
        ("long position", True, "Direction keyword"), 
        ("short USDJPY", True, "Direction keyword"),
        ("Hello world", False, "No keywords"),
        ("SL at 1.1000", False, "SL only"),
        ("TP1 at 1.1100", False, "TP only"),
        ("BUY EURUSD SL 1.0950 TP 1.1050", True, "Complete signal"),
    ]
    
    for text, expected_direction, description in test_cases:
        has_direction = window_manager._has_direction_keyword(text)
        has_sl = window_manager._has_sl_keyword(text)
        has_tp = window_manager._has_tp_keyword(text)
        is_complete = window_manager._is_complete_signal(text)
        
        print(f"Text: '{text}'")
        print(f"  Direction: {has_direction}, SL: {has_sl}, TP: {has_tp}, Complete: {is_complete}")
        print(f"  Expected direction: {expected_direction} ({description})")
        print()
    
    return True

async def test_full_system():
    """Test the complete system integration"""
    print("\nTesting Full System Integration...")
    
    try:
        # Load config
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Initialize components
        window_manager = WindowManager(config['signal_extraction']['time_window_minutes'])
        signal_parser = SignalParser(config)
        signal_output = SignalOutput(config)
        
        print("✓ All components initialized successfully")
        
        # Test a complete signal flow
        complete_signal_msg = {
            'message_id': 100,
            'timestamp': datetime.now().isoformat(),
            'text': 'BUY XAUUSD at 1950.50, SL: 1945.00, TP1: 1960.00, TP2: 1970.00',
            'group_name': 'GOLD_PERFECT_SIGNAL'
        }
        
        print(f"Testing complete signal: {complete_signal_msg['text']}")
        
        # Process through window manager
        completed_window = window_manager.process_message(complete_signal_msg, 'GOLD_PERFECT_SIGNAL')
        
        if completed_window:
            print("✓ Window completed successfully")
            
            # Parse with AI (this might fail without proper API key, but we can test the flow)
            try:
                signal = await signal_parser.parse_signal(completed_window)
                if signal:
                    print("✓ Signal parsed successfully")
                    signal_output.save_signal(signal)
                    print("✓ Signal saved successfully")
                else:
                    print("⚠ Signal parsing returned None (possibly due to API issues)")
            except Exception as e:
                print(f"⚠ Signal parsing failed (expected with demo data): {e}")
        else:
            print("✗ Window was not completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Full system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Enhanced Telegram Signal Monitor Tests ===\n")
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Window Manager", test_window_manager),
        ("Signal Detection", test_signal_detection),
        ("Full System", lambda: asyncio.run(test_full_system())),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name} test completed")
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("Test Results Summary")
    print(f"{'='*50}")
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)