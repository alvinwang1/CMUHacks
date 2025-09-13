import boto3
from boto3.dynamodb.conditions import Key  # for query later

region = "us-east-2"
table_name = "Order_Logs"

dynamodb = boto3.resource("dynamodb", region_name=region)
from decimal import Decimal

table = dynamodb.Table(table_name)
item = {
    "id": "1",
    "time": "2025-01-01 12:00:00",
    "product_id": "PROD-001",
    "transaction_type": "purchase",
    "quantity": 5,
    "unit_price": Decimal("12.99"),
    "total_price": Decimal("64.95"),
    "current_balance": Decimal("150.00")
}
table.put_item(Item=item)


print("Wrote item.")
