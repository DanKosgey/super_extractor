import os
import yaml
import google.generativeai as genai

def main():
    # Load configuration
    with open(r'C:\Users\PC\OneDrive\Desktop\python\Gemini extractor\gemini_signal_extractor\config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get API credentials from config
    api_key = config['gemini']['api_key']
    if not api_key:
        print("Error: GEMINI_API_KEY not found in config.yaml")
        return
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # List available models
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

def list_available_models():
    """List all available Gemini models and their capabilities."""
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return
        
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # List all models
        models = genai.list_models()
        
        print("\nAvailable Gemini Models:")
        print("=" * 50)
        
        for model in models:
            print(f"\nModel: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Generation Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    main() 