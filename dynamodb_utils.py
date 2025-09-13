# dynamodb_utils.py

import boto3
from botocore.exceptions import ClientError

from dynamodb_config import AWS_REGION, TABLE_NAMES

class DynamoDBManager:
    """
    Manages all interactions with the DynamoDB database for the vending machine agent.
    """
    def __init__(self):
        """Initializes the DynamoDB client."""
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            self.tables = {name: self.dynamodb.Table(table_name) for name, table_name in TABLE_NAMES.items()}
        except ClientError as e:
            print(f"Error initializing DynamoDB client: {e}")
            raise

    def get_current_stock_state(self, current_date):
        """Helper to get the single current 'is_actual' entry."""
        table = self.tables['stock']
        try:
            # Note: A scan is used for simplicity. For a large dataset, a GSI on 'is_actual' would be more efficient.
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('is_actual').eq(1) & boto3.dynamodb.conditions.Attr('time_of_day').eq('closing') # & boto3.dynamodb.conditions.Attr('time').begins_with(current_date)
            )
            assert len(response.get('Items', [])) <= 10, "More than 10 products found in the stock table"
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error scanning table Stock: {e}")
            raise
    
    def get_current_balance_state(self, current_date):
        """Helper to get the single current 'is_actual' entry."""
        table = self.tables['balance']
        try:
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('is_active').eq(1) & boto3.dynamodb.conditions.Attr('time_of_day').eq('closing') #& boto3.dynamodb.conditions.Attr('time').begins_with(current_date)
            )
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error scanning table Balance: {e}")
            raise

    def get_historical_events(self, current_date):
        """
        Fetches historical events that occurred on the same day.
        
        Args:
            current_date (str): The current date in 'YYYY-MM-DD' format.
        
        Returns:
            list: A list of DynamoDB items for the events of the current day.
        """
        table = self.tables['events']
        try:
            # A scan with a filter is used here. For large tables, a different
            # DynamoDB schema (e.g., using date as a partition key) would be
            # more performant. This approach is simple but less efficient.
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('time').begins_with(current_date)
            )
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error scanning table events: {e}")
            raise

    def get_supplier_info(self):
        """Fetches the buying prices for all products from the supplier."""
        table = self.tables['supplier']
        try:
            response = table.scan()
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error scanning table {TABLE_NAMES['supplier']}: {e}")
            raise
    
    def update_state(self, old_stock_items, new_stock_data, old_balance_items, new_balance_data):
        """
        Atomically updates the stock and balance tables.
        
        This method performs a multi-item update using a transaction. It first sets
        the old 'is_actual' flags to False, then inserts the new 'morning' state.
        """
        table_stock = self.tables['stock']
        table_balance = self.tables['balance']
        
        transaction_items = []
        
        # 1. Update old stock entries to is_actual=False
        for item in old_stock_items:
            transaction_items.append({
                'Update': {
                    'TableName': table_stock.name,
                    'Key': {'stock_id': item['stock_id']},
                    'UpdateExpression': 'SET is_actual = :f',
                    'ExpressionAttributeValues': {':f': 0}
                }
            })
            
        # 2. Update old balance entry to is_active=False
        for item in old_balance_items:
            transaction_items.append({
                'Update': {
                    'TableName': table_balance.name,
                    'Key': {'trans_id': item['trans_id']},
                    'UpdateExpression': 'SET is_active = :f',
                    'ExpressionAttributeValues': {':f': 0}
                }
            })

        # 2. Insert new stock entries
        for stock_item in new_stock_data:
            transaction_items.append({
                'Put': {
                    'TableName': table_stock.name,
                    'Item': stock_item
                }
            })

        # 3. Insert new balance entry
        transaction_items.append({
            'Put': {
                'TableName': table_balance.name,
                'Item': new_balance_data
            }
        })

        try:
            self.dynamodb.meta.client.transact_write_items(
                TransactItems=transaction_items
            )
            print("Database updated successfully with new stock and balance.")
        except ClientError as e:
            print(f"Transaction failed: {e}")
            raise