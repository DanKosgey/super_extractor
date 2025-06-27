import asyncio
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat
import os
from dotenv import load_dotenv
import yaml

async def list_groups():
    # Load environment variables for API credentials
    load_dotenv()
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    phone = os.getenv('TELEGRAM_PHONE')
    
    if not api_id or not api_hash:
        raise ValueError("TELEGRAM_API_ID or TELEGRAM_API_HASH not found in environment variables")
    
    # Initialize client
    client = TelegramClient('telegram_session', api_id, api_hash)

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

        print("\nAvailable groups and channels:")
        async for dialog in client.iter_dialogs():
            if isinstance(dialog.entity, (Channel, Chat)):
                print(f"- {dialog.name}")

    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(list_groups()) 