# Gemini Signal Extractor

## Overview
This project is a robust, modular pipeline for extracting, parsing, and validating trading signals from Telegram groups, with a focus on Forex, commodities, and crypto signals. It is designed for reliability, resumability, and extensibility, and leverages Google Gemini AI for advanced signal parsing.

---

## Directory Structure

```
.
├── config.yaml                # Main configuration file for all scripts
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── signal_extractor.py        # Extracts and windows raw messages into time-based signal windows
├── signal_parser.py           # Parses windows into structured trading signals using Gemini AI
├── telegram_extractor.py      # Downloads raw messages from Telegram groups
├── telegram_monitor.py        # Monitors Telegram groups for new messages
├── group_ids.txt              # List of Telegram group IDs
├── list_groups.py             # Utility: List available Telegram groups
├── list_models.py             # Utility: List available Gemini models
├── parser.py                  # (Legacy/utility) Parsing helpers
├── progress_tracker.py        # Tracks progress for resumable parsing
├── rate_limiter.py            # Handles API rate limiting for Gemini
├── windows.py                 # (Legacy/utility) Windowing helpers
├── data/
│   ├── groups/
│   │   └── <GROUP_NAME>/
│   │       ├── raw_messages.csv   # Raw Telegram messages for the group
│   │       ├── windows.csv        # Time-windowed message batches
│   │       └── signals.csv        # Parsed signals for the group
│   └── final_signals/
│       └── final_signals.csv      # All parsed signals across groups
├── logs/
│   └── *.log                  # Log files for each script
├── notebooks/                 # Jupyter notebooks for EDA and insights
├── tests/                     # Unit tests
├── utils/                     # Logging and helper utilities
```

---

## Main Scripts and Their Purpose

### 1. `telegram_extractor.py`
- **Purpose:** Downloads raw messages from specified Telegram groups using the Telegram API.
- **Output:** `data/groups/<GROUP_NAME>/raw_messages.csv`
- **Config:** Uses `config.yaml` for group IDs and API credentials.

### 2. `signal_extractor.py`
- **Purpose:** Reads raw messages and creates time-based windows (batches) of messages likely to contain a trading signal.
- **Logic:**
  - Groups messages into windows based on time and signal completeness.
  - Each window is saved as a row in `windows.csv`.
- **Output:** `data/groups/<GROUP_NAME>/windows.csv`

### 3. `signal_parser.py`
- **Purpose:** Parses each window using Gemini AI to extract structured trading signals.
- **Features:**
  - Resumable: Tracks progress and can resume from where it left off.
  - Merges all columns from `windows.csv` with parsed signal fields in output.
  - Handles batching, rate limiting, and error logging.
- **Output:**
  - `data/groups/<GROUP_NAME>/signals.csv` (per-group signals)
  - `data/final_signals/final_signals.csv` (all signals across groups)

### 4. `progress_tracker.py`
- **Purpose:** Tracks which windows have been processed for each group, enabling safe interruption and resumption.
- **Output:** `data/processing_state.json`

### 5. `rate_limiter.py`
- **Purpose:** Ensures Gemini API calls do not exceed configured rate limits.
- **Configurable:** Reads limits from `config.yaml`.

### 6. `telegram_monitor.py`
- **Purpose:** Monitors Telegram groups for new messages in real time.
- **Output:** Appends new messages to `raw_messages.csv`.

### 7. `list_groups.py`, `list_models.py`
- **Purpose:** Utility scripts to list available Telegram groups and Gemini models.

### 8. `notebooks/`
- **Purpose:** Contains Jupyter notebooks for exploratory data analysis and signal insights.

### 9. `tests/`
- **Purpose:** Unit tests for core modules.

### 10. `utils/`
- **Purpose:** Logging configuration and helper utilities.

---

## Data Flow
1. **Raw Extraction:** `telegram_extractor.py` → `raw_messages.csv`
2. **Windowing:** `signal_extractor.py` → `windows.csv`
3. **Parsing:** `signal_parser.py` → `signals.csv` (per group) and `final_signals.csv` (all groups)

---

## Configuration (`config.yaml`)
- Centralizes all settings: paths, Telegram credentials, Gemini API keys, rate limits, batching, and signal extraction rules.
- **Edit this file to:**
  - Add/remove groups
  - Change data/log locations
  - Adjust rate limits and batching
  - Update signal extraction rules

---

## Logging
- All scripts log to `logs/` with detailed info and error messages.
- Logging level and format are configurable in `config.yaml`.

---

## Resumability & Reliability
- Progress is tracked in `data/processing_state.json`.
- You can safely interrupt and resume parsing at any time.
- All outputs are atomic and robust to crashes.

---

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies (e.g., `pandas`, `pyyaml`, `telethon`, `google-generativeai`)

---

## Getting Started
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure `config.yaml`** with your Telegram and Gemini API credentials and desired groups.
3. **Run extraction and parsing:**
   ```sh
   python telegram_extractor.py
   python signal_extractor.py
   python signal_parser.py
   ```
4. **Check outputs:**
   - Per-group: `data/groups/<GROUP_NAME>/signals.csv`
   - All signals: `data/final_signals/final_signals.csv`

---

## Notes
- All outputs include both the original window data and parsed signal fields for full traceability.
- The pipeline is modular: you can run each step independently or automate the full flow.
- For troubleshooting, check the logs in the `logs/` directory.

---

## License
This project is for educational and research purposes. Use at your own risk.
