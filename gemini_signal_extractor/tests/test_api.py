import os
import yaml
import google.generativeai as genai

def list_available_models():
    """List all available Gemini models."""
    # Load API key from config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            api_key = config['gemini']['api_key']
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return
        
    if not api_key:
        print("Error: Gemini API key not found in config.yaml")
        return
        
    # Configure the API
    genai.configure(api_key=api_key)
    
    try:
        # List all models
        print("\nAvailable Gemini Models:")
        print("=" * 50)
        
        models = genai.list_models()
        for model in models:
            print(f"\nModel: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print(f"Generation Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("1. Check if your API key is valid")
        print("2. Ensure you have an active internet connection")
        print("3. Verify that you have the latest version of google-generativeai installed")
        print("4. Check if the model you're trying to use is available in your region")

if __name__ == "__main__":
    list_available_models() 