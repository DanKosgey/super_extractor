import os
import yaml
import logging
import asyncio
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import InputPeerChannel

class TelegramExtractor:
    def __init__(self, config_path: Optional[str] = None):
        # Load env (.env) first, so config.yaml can reference env vars if needed
        load_dotenv()

        # Locate config.yaml
        if config_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base, 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Prepare directories
        base = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.join(base, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        self.data_dir = os.path.join(base, 'data', 'groups')
        os.makedirs(self.data_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.logs_dir, 'telegram_extractor.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Telegram credentials
        tg_cfg = self.config['telegram']
        self.api_id = tg_cfg['api_id']
        self.api_hash = tg_cfg['api_hash']
        self.phone = tg_cfg['phone']

        # Groups
        # group_ids: mapping name→numeric ID; groups: list of names to extract
        self.group_ids = tg_cfg.get('group_ids', {})
        self.active_groups = tg_cfg.get('groups', [])

        # Optional total count limit per group
        self.total_count_limit = tg_cfg.get('total_count_limit', 0)
        
        # Initialize client
        self.client = TelegramClient('telegram_session', self.api_id, self.api_hash)
        
    async def connect(self):
        await self.client.connect()
        if not await self.client.is_user_authorized():
            await self.client.send_code_request(self.phone)
            code = input('Enter Telegram code: ')
            await self.client.sign_in(self.phone, code)
        self.logger.info("Connected to Telegram")

    async def get_group_entity(self, group_name: str):
        """
        Resolve a group/channel by exact name key or by substring search.
        Returns the entity or None.
        """
        try:
            # Method 1: If numeric ID is provided in config, use it directly
            if group_name in self.group_ids:
                channel_id = self.group_ids[group_name]
                self.logger.info(f"Attempting to get entity for channel ID: {channel_id}")
                
                # Convert to negative ID format if it's a channel/supergroup
                if channel_id > 0:
                    # For channels/supergroups, convert to negative format
                    negative_id = -1000000000000 - channel_id
                    self.logger.info(f"Converted ID {channel_id} to {negative_id}")
                    return await self.client.get_entity(negative_id)
                else:
                    return await self.client.get_entity(channel_id)

            # Method 2: Try fuzzy match on config keys
            for name, cid in self.group_ids.items():
                if group_name.lower() in name.lower():
                    if cid > 0:
                        negative_id = -1000000000000 - cid
                        return await self.client.get_entity(negative_id)
                    else:
                        return await self.client.get_entity(cid)

            # Method 3: Try by username/invite link
            if group_name.startswith('https://t.me/'):
                return await self.client.get_entity(group_name)
            elif group_name.startswith('@'):
                return await self.client.get_entity(group_name)
            else:
                # Try adding @ prefix
                return await self.client.get_entity(f'@{group_name}')
                
        except Exception as e:
            self.logger.error(f"Could not resolve '{group_name}': {e}")
            
            # Additional debugging info
            if group_name in self.group_ids:
                self.logger.error(f"Channel ID from config: {self.group_ids[group_name]}")
                
        return None

    async def extract_messages(self, group_name: str) -> pd.DataFrame:
        """Fetch all messages (up to limit) from a single group and return a sorted DataFrame."""
        self.logger.info(f"Starting extraction for '{group_name}'")
        entity = await self.get_group_entity(group_name)
        if entity is None:
            self.logger.error(f"Could not get entity for '{group_name}', skipping...")
            return pd.DataFrame()

        self.logger.info(f"Successfully got entity for '{group_name}': {entity}")

        # Convert entity to input peer
        input_peer = await self.client.get_input_entity(entity)

        # Prepare output directory for this group
        group_dir = os.path.join(self.data_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        all_msgs = []
        offset_id = 0
        fetch_limit = 100

        try:
            while True:
                history = await self.client(GetHistoryRequest(
                    peer=input_peer,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=fetch_limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
                msgs = history.messages
                if not msgs:
                    break

                all_msgs.extend(msgs)
                offset_id = msgs[-1].id

                self.logger.info(f"Fetched {len(msgs)} messages, total: {len(all_msgs)}")

                # Stop if we've reached user-configured limit
                if self.total_count_limit and len(all_msgs) >= self.total_count_limit:
                    all_msgs = all_msgs[:self.total_count_limit]
                    break

                # If fewer than fetch_limit returned, we've fetched all
                if len(msgs) < fetch_limit:
                    break
        except Exception as e:
            self.logger.error(f"Error fetching messages from '{group_name}': {e}")
            return pd.DataFrame()

        # Build DataFrame
        data = []
        for m in all_msgs:
            if hasattr(m, 'message') and m.message:
                data.append({
                    'message_id': m.id,
                    'date': m.date,
                    'message': m.message,
                    'from_id': getattr(m, 'from_id', None),
                    'reply_to': getattr(m, 'reply_to', None)
                })
        
        df = pd.DataFrame(data)

        if not df.empty:
            # Sort by date (oldest first) for chronological order
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            # Save
            out_csv = os.path.join(group_dir, 'raw_messages.csv')
            df.to_csv(out_csv, index=False, encoding='utf-8')
            self.logger.info(f"Saved {len(df)} messages to {out_csv} (sorted chronologically)")
        else:
            self.logger.warning(f"No messages found for '{group_name}'")
        
        return df

    async def process_all_groups(self):
        for group in self.active_groups:
            try:
                await self.extract_messages(group)
            except Exception as e:
                self.logger.error(f"Failed processing '{group}': {e}")

    async def run(self):
        try:
            await self.connect()
            await self.process_all_groups()
        finally:
            await self.client.disconnect()
            self.logger.info("Disconnected from Telegram")

def main():
    extractor = TelegramExtractor()
    asyncio.run(extractor.run())

if __name__ == '__main__':
    main()