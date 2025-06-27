import os
import yaml
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'prompt_test.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info("Successfully loaded config.yaml")
            return config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {str(e)}")
        raise

def load_prompt():
    try:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'extract_signal_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            logger.info("Successfully loaded prompt file")
            return prompt
    except Exception as e:
        logger.error(f"Error loading prompt file: {str(e)}")
        raise

def test_signal(model, prompt_template, test_text):
    prompt = prompt_template.format(text=test_text)
    try:
        logger.info(f"\nTesting signal:\n{test_text}\n")
        logger.info("Sending request to Gemini API...")
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini API")
        logger.info(f"Raw response:\n{response.text}\n")
        
        try:
            # Try to parse JSON response
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                signal_json = response.text[json_start:json_end]
                signal = json.loads(signal_json)
                logger.info(f"Successfully parsed JSON:\n{json.dumps(signal, indent=2)}\n")
            else:
                logger.warning("No JSON found in response\n")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}\n")
    except Exception as e:
        logger.error(f"Error during API call: {str(e)}\n")

def main():
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
        
        # Load configuration and setup
        logger.info("Loading configuration...")
        config = load_config()
        
        api_key = config['gemini']['api_key']
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config.yaml")
        logger.info("API key found in config")
        
        # Initialize Gemini
        logger.info("Initializing Gemini API...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config['gemini']['model'])
        logger.info(f"Using model: {config['gemini']['model']}")
        
        # Load prompt template
        logger.info("Loading prompt template...")
        prompt_template = load_prompt()
        
        # Test cases
        test_cases = [
            # Test Case 1: Standard GOLD signal
            """gold sell now
2159-2161
sl 2165
Tp1 2155.20""",
            
            # Test Case 2: Multiple TPs
            """EURUSD BUY
Entry: 1.0850-1.0860
SL: 1.0820
TP1: 1.0900
TP2: 1.0950
TP3: 1.1000""",
            
            # Test Case 3: Different format
            """XAUUSD
SELL
Entry Zone: 2300-2305
Stop Loss: 2310
Take Profit 1: 2290
Take Profit 2: 2280""",
            
            # Test Case 4: Mixed case and spacing
            """Gold
Buy
entry 2150-2155
sl 2145
tp1 2160
tp2 2165""",
            
            # Test Case 5: Invalid symbol
            """BTCUSD buy
Entry: 50000
SL: 49000
TP1: 52000""",
            
            # Test Case 6: Missing components
            """TP1 hit! Great trade!""",
            
            # Test Case 7: Real-world format with emojis
            """🎯 GOLD SELL NOW
Entry: 2159-2161
SL: 2165
TP1: 2155.20
TP2: 2150
Good luck! 🚀""",
            
            # Test Case 8: Long/Short format
            """EURUSD
LONG
Entry: 1.0850
SL: 1.0820
TP1: 1.0900""",
            
            # Test Case 9: Complex format with extra text
            """Hey traders! Here's a new signal:
GOLD SELL
Entry Zone: 2159-2161
Stop Loss: 2165
Take Profit 1: 2155.20
Take Profit 2: 2150
Remember to manage your risk!"""
        ]
        
        # Run tests
        logger.info("Starting prompt tests...")
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Test Case {i}")
            logger.info(f"{'='*50}")
            test_signal(model, prompt_template, test_case)
            logger.info(f"{'='*50}\n")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 