import os
import yaml
import google.generativeai as genai
import logging

def setup_logging():
    """Setup logging with UTF-8 encoding."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'test_model.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_model():
    """Test the Gemini model with a simple prompt."""
    logger = setup_logging()
    logger.info("Starting Gemini model test...")
    
    # Load API key from config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            api_key = config['gemini']['api_key']
            model_name = config['gemini']['model']
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return False
        
    if not api_key:
        logger.error("Error: Gemini API key not found in config.yaml")
        return False
        
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40
            }
        )
        
        # Test prompt
        test_prompt = """
        Extract trading signal from this text:
        Buy XAUUSD
        Entry: 2150-2155
        SL: 2145
        TP1: 2160
        TP2: 2165
        """
        
        print("\nTesting model with prompt:")
        print("=" * 50)
        print(test_prompt)
        print("=" * 50)
        
        # Generate response
        response = model.generate_content(test_prompt)
        print("\nModel response:")
        print("=" * 50)
        print(response.text)
        print("=" * 50)
        return True
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("1. Check if your API key is valid")
        print("2. Ensure you have an active internet connection")
        print("3. Verify that you have the latest version of google-generativeai installed")
        print("4. Check if the model you're trying to use is available in your region")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!") 