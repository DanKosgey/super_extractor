#!/usr/bin/env python3
"""
Enhanced Telegram Signal Monitor - Usage Example

This script demonstrates how to run the enhanced Telegram signal monitoring system.

The system will:
1. Monitor specified Telegram groups for new messages
2. Detect direction keywords (buy, sell, long, short)
3. If complete signal (direction + SL + TP) -> send immediately to AI
4. If partial signal (direction only) -> open 2-minute window to collect more messages
5. Save all parsed signals to CSV files
"""

import asyncio
import logging
import sys
from pathlib import Path
from telegram_monitor import LiveTelegramMonitor

def setup_logging():
    """Setup basic logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('telegram_monitor_demo.log')
        ]
    )

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🚀 ENHANCED TELEGRAM SIGNAL MONITOR")
    print("=" * 70)
    print()
    print("Features:")
    print("✓ Real-time Telegram message monitoring")
    print("✓ Intelligent signal window management")
    print("✓ Direction keyword detection (buy, sell, long, short)")
    print("✓ SL/TP completion checking")
    print("✓ 2-minute waiting windows for partial signals")
    print("✓ AI-powered signal parsing with Gemini")
    print("✓ CSV output for all signals")
    print()
    print("Workflow:")
    print("1. Monitor configured Telegram groups")
    print("2. Detect messages with direction keywords")
    print("3. If direction + SL + TP found -> parse immediately")
    print("4. If only direction found -> wait 2 minutes for SL/TP")
    print("5. Parse complete signals with AI and save to CSV")
    print()
    print("=" * 70)
    print()

def check_config():
    """Check if configuration is properly set"""
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("❌ ERROR: config.yaml not found!")
        print("Please ensure config.yaml exists with:")
        print("- Telegram API credentials")
        print("- Group IDs to monitor")
        print("- Gemini API key")
        return False
    
    print("✓ Configuration file found")
    return True

def print_usage_info():
    """Print usage information"""
    print("📝 USAGE INFORMATION:")
    print()
    print("1. CONFIGURATION:")
    print("   - Edit config.yaml to set your API credentials")
    print("   - Add Telegram group IDs to monitor")
    print("   - Set Gemini API key for signal parsing")
    print()
    print("2. SIGNAL FORMAT:")
    print("   Complete signals should contain:")
    print("   - Direction: buy, sell, long, short")
    print("   - Stop Loss: sl, stop loss") 
    print("   - Take Profit: tp, tp1, tp2, take profit, target")
    print()
    print("   Examples:")
    print("   ✓ 'BUY EURUSD at 1.1000, SL: 1.0950, TP1: 1.1050'")
    print("   ✓ 'SELL GOLD 1950, stop loss 1955, target 1940'")
    print("   ✓ 'Long GBPUSD SL 1.2500 TP 1.2600'")
    print()
    print("3. OUTPUT:")
    print("   - Live signals: data/signals/live_signals.csv")
    print("   - Group-specific: data/groups/{GROUP_NAME}/live_signals.csv")
    print("   - Raw messages: data/groups/{GROUP_NAME}/raw_messages.csv")
    print()

async def main():
    """Main function"""
    setup_logging()
    print_banner()
    
    # Check configuration
    if not check_config():
        print_usage_info()
        return 1
    
    try:
        # Initialize the monitor
        print("🔧 Initializing Telegram Signal Monitor...")
        monitor = LiveTelegramMonitor()
        
        print("✓ Monitor initialized successfully")
        print()
        print("🎯 MONITORING ACTIVE - The system is now:")
        print("   • Connecting to Telegram...")
        print("   • Monitoring configured groups for new messages")
        print("   • Detecting trading signals automatically")
        print("   • Parsing complete signals with AI")
        print("   • Saving results to CSV files")
        print()
        print("📊 Watch the logs below for real-time activity:")
        print("=" * 70)
        
        # Start monitoring
        await monitor.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logging.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)