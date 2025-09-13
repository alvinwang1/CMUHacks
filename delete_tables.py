import boto3
from botocore.exceptions import BotoCoreError, ClientError

region = "us-east-2"
dynamodb = boto3.resource("dynamodb", region_name=region)

events_table  = dynamodb.Table("events_test")   # PK: event_id
stock_table   = dynamodb.Table("stock_test")    # PK: stock_id
balance_table = dynamodb.Table("balance_test")  # PK: trans_id

def delete_all(table, pk_name: str, sk_name: str | None = None, progress_every: int = 100) -> int:
    """Delete all items from `table` using primary key attributes."""
    print(f"\nScanning '{table.name}' for items to delete...")
    items_to_delete = []
    last_evaluated_key = None

    # Build projection for keys only
    expr_names = {"#pk": pk_name}
    proj = "#pk"
    if sk_name:
        expr_names["#sk"] = sk_name
        proj = proj + ", #sk"

    try:
        while True:
            scan_kwargs = {"ProjectionExpression": proj, "ExpressionAttributeNames": expr_names}
            if last_evaluated_key:
                scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

            resp = table.scan(**scan_kwargs)
            items_to_delete.extend(resp.get("Items", []))
            last_evaluated_key = resp.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break

        total = len(items_to_delete)
        print(f"Found {total} item(s) in '{table.name}'.")

        if total == 0:
            return 0

        print(f"Starting batch deletion on '{table.name}'...")
        deleted = 0
        with table.batch_writer() as batch:
            for it in items_to_delete:
                key = {pk_name: it[pk_name]}
                if sk_name:
                    key[sk_name] = it[sk_name]
                batch.delete_item(Key=key)
                deleted += 1
                if deleted % progress_every == 0:
                    print(f"Deleted {deleted}/{total}...")

        print(f"Successfully deleted {deleted} item(s) from '{table.name}'.")
        return deleted

    except (BotoCoreError, ClientError) as e:
        print(f"Error deleting from '{table.name}': {e}")
        return 0

def delete_all_tables():
    delete_all(balance_table, pk_name="trans_id")
    delete_all(events_table,  pk_name="event_id")
    delete_all(stock_table,   pk_name="stock_id")

if __name__ == "__main__":
    # Delete everything from each table
    delete_all()