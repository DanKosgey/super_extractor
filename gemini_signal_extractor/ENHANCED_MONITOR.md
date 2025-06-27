# Enhanced Telegram Signal Monitor

## Overview

The Enhanced Telegram Signal Monitor is a sophisticated real-time system that monitors Telegram groups for trading signals with perfect accuracy and timing. It implements the exact workflow you specified for signal detection and processing.

## Key Features

### 🎯 Perfect Signal Detection
- **Direction Keywords**: Detects buy, sell, long, short
- **Stop Loss Detection**: Recognizes sl, stop loss, stop-loss, stoploss
- **Take Profit Detection**: Identifies tp, tp1, tp2, tp3, tp4, take profit, target
- **Complete Signal Recognition**: Instantly identifies signals with all components

### ⏱️ Intelligent Window Management
- **Immediate Processing**: Complete signals (direction + SL + TP) are sent to AI immediately
- **Smart Waiting**: Direction-only messages trigger 2-minute waiting windows
- **Perfect Timing**: Collects additional messages within the time window
- **Completion Detection**: Automatically detects when SL/TP are found
- **Clean Expiry**: Discards incomplete signals after window timeout

### 🤖 AI-Powered Parsing
- **Gemini AI Integration**: Uses Google's Gemini AI for signal extraction
- **Structured Output**: Extracts symbol, direction, entry, SL, TP1-4, confidence
- **Validation**: Ensures price relationships are correct
- **Error Handling**: Robust error handling and retry logic

### 📊 Comprehensive Output
- **CSV Files**: All signals saved to structured CSV files
- **Group Separation**: Separate files per group and combined file
- **Raw Messages**: Backup of all original messages
- **Real-time Logging**: Detailed logs of all activities

## How It Works

### 1. Message Monitoring
```
New Telegram Message → Check for Direction Keywords
```

### 2. Signal Classification
```
Has Direction Keywords?
├─ Yes → Check for SL and TP
│   ├─ Has All (Direction + SL + TP) → Complete Signal → Send to AI Immediately
│   └─ Has Only Direction → Create 2-Minute Window → Wait for SL/TP
└─ No → Ignore Message
```

### 3. Window Management
```
2-Minute Window Created → Collect New Messages → Check Each Message
├─ Message Contains SL/TP → Complete Window → Send to AI → Close Window
└─ Window Expires → Discard → Log as Incomplete
```

### 4. AI Processing
```
Complete Signal → Gemini AI → Extract Structured Data → Validate → Save to CSV
```

## Configuration

### Essential Settings (config.yaml)
```yaml
telegram:
  api_id: YOUR_API_ID
  api_hash: "YOUR_API_HASH"
  phone: "+YOUR_PHONE"
  groups:
    - GOLD_PERFECT_SIGNAL
    - SNIPER_FX_SIGNAL

gemini:
  api_key: "YOUR_GEMINI_API_KEY"
  model: "gemini-1.5-flash"

signal_extraction:
  time_window_minutes: 2  # The 't' minutes you specified
```

## File Structure
```
data/
├── signals/
│   └── live_signals.csv          # All signals combined
├── groups/
│   └── {GROUP_NAME}/
│       ├── raw_messages.csv      # Original messages
│       └── live_signals.csv      # Group-specific signals
└── logs/
    └── telegram_monitor.log      # System logs
```

## Usage

### Basic Usage
```bash
cd /app/gemini_signal_extractor
python run_monitor.py
```

### Testing
```bash
python test_monitor.py
```

### Manual Monitoring
```bash
python telegram_monitor.py
```

## Signal Examples

### Complete Signal (Processed Immediately)
```
"BUY EURUSD at 1.1000, SL: 1.0950, TP1: 1.1050, TP2: 1.1100"
```
→ Detected as complete → Sent to AI immediately → Parsed and saved

### Partial Signal (Creates Window)
```
Message 1 (10:00): "BUY EURUSD at 1.1000"
Message 2 (10:01): "SL: 1.0950"  
Message 3 (10:01): "TP1: 1.1050"
```
→ Window created at 10:00 → Completed at 10:01 → Sent to AI → Saved

### Incomplete Signal (Discarded)
```
Message 1 (10:00): "BUY EURUSD at 1.1000"
... 2 minutes pass ...
Message 2 (10:02): "Good luck everyone!"
```
→ Window created at 10:00 → Expired at 10:02 → Discarded (no SL/TP found)

## Output Format

### CSV Structure
```csv
group_name,window_id,timestamp,symbol,direction,entry,sl,tp1,tp2,tp3,tp4,confidence,valid,corrections,validation_notes,context,raw_text
GOLD_PERFECT_SIGNAL,GOLD_1234567890_abcd1234,2024-01-01T10:00:00,XAUUSD,BUY,1950.50,1945.00,1960.00,1970.00,,,0.95,true,,Valid signal structure,Window: GOLD_1234567890_abcd1234 Messages: 1,"BUY XAUUSD at 1950.50 SL 1945.00 TP1 1960.00 TP2 1970.00"
```

## Performance & Reliability

- **Zero Message Loss**: All messages saved to raw CSV immediately
- **Crash Recovery**: State persistence for resumability  
- **Rate Limiting**: Respects API limits for Gemini
- **Background Processing**: Non-blocking message handling
- **Memory Management**: Automatic cleanup of old windows
- **Error Recovery**: Continues monitoring despite individual failures

## Monitoring & Debugging

### Real-time Logs
- New message detection
- Window creation and completion
- Signal parsing results
- Error conditions

### Log Levels
- **INFO**: Normal operation status
- **DEBUG**: Detailed processing steps
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures

## Requirements Met

✅ **Fetches new messages** from configured groups  
✅ **Direction keyword detection** with immediate processing  
✅ **Complete signal detection** (direction + SL + TP)  
✅ **2-minute window management** for partial signals  
✅ **TP1, TP2, etc. detection** as TP keywords  
✅ **Perfect timing accuracy** to avoid missing signals  
✅ **CSV output** for all signals and messages  
✅ **Robust error handling** and state management  

## Advanced Features

- **Multi-group support**: Monitor multiple groups simultaneously
- **Symbol validation**: Only processes valid trading symbols
- **Price validation**: Ensures correct price relationships
- **Confidence scoring**: AI provides confidence levels
- **State persistence**: Resume from crashes without data loss
- **Background cleanup**: Automatic memory management