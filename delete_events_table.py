import boto3

region = "us-east-2"
table_name = "events"

dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table(table_name)

print(f"Attempting to delete all items from table: {table_name} in region: {region}")

items_to_delete = []
last_evaluated_key = None

try:
    # Scan the table to get all primary keys (event_id)
    # This is necessary to know which items to delete, as delete_item requires the full primary key.
    print("Scanning table for items to delete...")
    while True:
        scan_kwargs = {
            "ProjectionExpression": "event_id" # Only retrieve the primary key 'event_id'
        }
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        response = table.scan(**scan_kwargs)
        items_to_delete.extend(response.get('Items', []))
        last_evaluated_key = response.get('LastEvaluatedKey')

        if not last_evaluated_key:
            break

    total_items_found = len(items_to_delete)
    print(f"Found {total_items_found} items to delete.")

    if total_items_found == 0:
        print("No items found to delete. Exiting.")
    else:
        deleted_count = 0
        print("Starting batch deletion using batch_writer...")
        # Use batch_writer for efficient deletion. It handles sending items in batches
        # and retrying unprocessed items automatically.
        with table.batch_writer() as batch_writer:
            for item in items_to_delete:
                batch_writer.delete_item(Key={'event_id': item['event_id']})
                deleted_count += 1
                if deleted_count % 100 == 0: # Provide progress updates
                    print(f"Deleted {deleted_count}/{total_items_found} items...")

        print(f"Successfully deleted {deleted_count} items from table '{table_name}'.")

except Exception as e:
    print(f"An error occurred during deletion: {e}")
