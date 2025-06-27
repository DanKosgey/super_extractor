import asyncio
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat
import os
from dotenv import load_dotenv
import yaml
import logging
from telethon.tl.functions.messages import GetHistoryRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_groups():
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get API credentials from config
    api_id = config['TELEGRAM_API_ID']
    api_hash = config['TELEGRAM_API_HASH']
    phone = config['TELEGRAM_PHONE']
    
    if not api_id or not api_hash:
        raise ValueError("TELEGRAM_API_ID or TELEGRAM_API_HASH not found in config.yaml")
    
    # Initialize client
    client = TelegramClient('test_session', api_id, api_hash)
    
    try:
        print("\nConnecting to Telegram...")
        await client.connect()
        
        if not await client.is_user_authorized():
            print(f"\nSending code to {phone}...")
            await client.send_code_request(phone)
            print("\nPlease check your Telegram app for the code.")
            code = input("Enter the code you received: ")
            await client.sign_in(phone, code)
            print("Successfully signed in!")
        else:
            print("Already authorized!")
        
        print("\nFetching all available groups...")
        available_groups = {}
        async for dialog in client.iter_dialogs():
            if isinstance(dialog.entity, (Channel, Chat)):
                group_name = dialog.name
                group_id = dialog.id
                group_type = "Channel" if isinstance(dialog.entity, Channel) else "Chat"
                available_groups[group_name] = {
                    'id': group_id,
                    'type': group_type,
                    'entity': dialog.entity
                }
                print(f"\nFound group: {group_name}")
                print(f"  ID: {group_id}")
                print(f"  Type: {group_type}")
                print(f"  Username: {getattr(dialog.entity, 'username', 'None')}")
        
        print("\nValidating groups from config.yaml:")
        config_groups = config['telegram']['groups']
        valid_groups = []
        invalid_groups = []
        
        for group in config_groups:
            if group in available_groups:
                valid_groups.append(group)
                print(f"\n✅ Valid group: {group}")
                print(f"  ID: {available_groups[group]['id']}")
                print(f"  Type: {available_groups[group]['type']}")
            else:
                invalid_groups.append(group)
                print(f"\n❌ Invalid group: {group}")
                # Try to find similar names
                similar = [g for g in available_groups.keys() if group.lower() in g.lower()]
                if similar:
                    print("  Similar groups found:")
                    for s in similar:
                        print(f"    - {s}")
        
        print("\nSummary:")
        print(f"Total groups in config: {len(config_groups)}")
        print(f"Valid groups: {len(valid_groups)}")
        print(f"Invalid groups: {len(invalid_groups)}")
        
        # Save valid groups to a new config file
        if valid_groups:
            new_config = config.copy()
            new_config['telegram']['groups'] = valid_groups
            config_valid_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_valid_groups.yaml')
            with open(config_valid_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, allow_unicode=True)
            print("\nSaved valid groups to 'config_valid_groups.yaml'")
            
            # Also save group IDs for reference
            group_ids_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'group_ids.txt')
            with open(group_ids_path, 'w', encoding='utf-8') as f:
                f.write("Group IDs for reference:\n")
                for group in valid_groups:
                    f.write(f"{group}: {available_groups[group]['id']}\n")
            print("Saved group IDs to 'group_ids.txt'")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_groups()) 