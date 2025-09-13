# vending_agent.py

import json
from datetime import datetime, timedelta
import uuid

from dynamodb_utils import DynamoDBManager
from prompt_builder import build_prompt
from llm_client import get_llm_client
from prompt_builder import DecimalEncoder
from decimal import Decimal
import time

class VendingAgent:
    """
    The AI agent responsible for the vending machine restocking decisions.
    """
    def __init__(self, init_budget=1000):
        """Initializes the agent with a DynamoDB manager and an LLM client."""
        self.db_manager = DynamoDBManager()
        self.llm_client = get_llm_client()
        self.initial_budget = init_budget

    def _get_llm_decision(self, prompt):
        """Sends the prompt to the LLM and parses the response."""
        try:
            response = self.llm_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # The LLM's response content is a list of content blocks.
            # We're interested in the text content, which should be the JSON string.
            json_response_str = response.content[0].text
            # print(json_response_str)
            json_start = json_response_str.find('```json')
            if json_start != -1:
                # Adjust start to skip the `json` marker
                json_start += len('```json')
                json_end = json_response_str.find('```', json_start)
                
                # Extract the JSON string only
                if json_end != -1:
                    json_string_to_parse = json_response_str[json_start:json_end].strip()
                else:
                    # If the end marker isn't found, assume the rest is the JSON
                    json_string_to_parse = json_response_str[json_start:].strip()
            else:
                # If no markers are found, assume the whole response is the JSON
                json_string_to_parse = json_response_str.strip()
            decision = json.loads(json_string_to_parse)
            print("LLM Decision (JSON):")
            print(json.dumps(decision, indent=2))
            return decision
        except Exception as e:
            print(f"Error getting or parsing LLM response: {e}")
            return None

    def _calculate_restock_cost(self, restock_plan, supplier_info):
        """Calculates the total cost of the restock plan."""
        supplier_prices = {item['product_name']: item['price'] for item in supplier_info}
        total_cost = 0
        for item in restock_plan:
            product_name = item['product_name']
            quantity = item['quantity_to_buy']
            cost_per_unit = supplier_prices.get(product_name)
            if cost_per_unit is None:
                print(f"Warning: Product '{product_name}' not found in supplier list. Skipping.")
                continue
            total_cost += cost_per_unit * quantity
        return total_cost

    def _prepare_data_for_update(self, old_stock, restock_plan, old_balance, supplier_info, new_date):
        """
        Prepares the data for the database update transaction based on the LLM's restock plan.
        Calculates new stock levels and balance.
        """
        # The LLM's response now provides the selling price directly in the restock_plan
        # so we no longer need a separate new_prices dictionary.

        # Convert old stock list to a dictionary for faster lookups
        old_stock_dict = {item['product_name']: item['quantity'] for item in old_stock}

        new_stock_data = []
        restocked_products = set()

        # 1. Process the restock plan from the LLM
        for item in restock_plan:
            product_name = item['product_name']
            restocked_products.add(product_name)

            # Ensure quantity_to_buy is a number and handle potential errors
            try:
                qty_to_buy = int(item.get('quantity_to_buy', 0))
            except (ValueError, TypeError):
                qty_to_buy = 0
                print(f"Warning: Quantity for '{product_name}' is not a valid number. Setting to 0.")

            # Ensure selling_price is a number and handle potential errors
            try:
                selling_price = Decimal(str(item.get('selling_price')))
            except (ValueError, TypeError):
                selling_price = None
                print(f"Warning: Selling price for '{product_name}' is not a valid number. Skipping item.")
            
            if selling_price is not None:
                # Calculate the new quantity by combining old stock and restock plan
                current_qty = old_stock_dict.get(product_name, 0)
                new_qty = current_qty + qty_to_buy # Max capacity is 10
                print(product_name, new_qty, current_qty, qty_to_buy)

                assert new_qty <= 10, "Selling qty > 10"
                
                if new_qty > 0:
                    new_stock_data.append({
                        'stock_id': str(uuid.uuid4()),
                        'product_name': product_name,
                        'quantity': new_qty,
                        'is_actual': 1,
                        'date': new_date,
                        'time_of_day': 'closing',
                        'selling_price': selling_price
                    })

        # 2. Handle products that were in the machine but not in the restock plan
        # These are kept if they have remaining stock and are not being removed.
        # My prompt tells me that the model can chose to discard them which means that if they don't appear in the restock plan, then they will not exist.
        
        # 3. Calculate new balance based on the restock plan cost
        total_cost = self._calculate_restock_cost(restock_plan, supplier_info)
        print(type(old_balance), type(total_cost))
        new_balance = float(old_balance) - float(total_cost)

        if new_balance < 0:
            raise ValueError("Restock plan exceeds available balance. Aborting transaction.")
            
        new_balance_data = {
            'trans_id': str(uuid.uuid4()),
            'balance': str(new_balance),
            'is_active': 1,
            'date': new_date,
            'time_of_day': 'opening'
        }

        return new_stock_data, new_balance_data

    def run_restock_cycle(self, date):
        """
        Main method to execute the full restocking cycle.
        """
        print(f"[{date}] Starting vending machine restock cycle...")

        t0 = time.time()
        # 1. Fetch data from DynamoDB
        try:
            current_stock = self.db_manager.get_current_stock_state(date)
            current_balance_items = self.db_manager.get_current_balance_state(date)
            historical_events = self.db_manager.get_historical_events(date)
            supplier_info = self.db_manager.get_supplier_info()
            
            current_balance = current_balance_items[0]['balance'] if current_balance_items else self.initial_budget
            
            print("Data fetched successfully.")
            print(f"Current Balance: ${current_balance}")
            print("Current Stock:")
            print(json.dumps(current_stock, indent=2, cls=DecimalEncoder))
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            return
        t1 = time.time()
        print('Time to load data: ', t1 - t0)

        # 2. Build the prompt for the LLM
        prompt = build_prompt(
            current_state={'stock': current_stock, 'balance': current_balance},
            historical_events=historical_events,
            supplier_info=supplier_info,
            current_date=date
        )
        t2 = time.time()
        print('Time to Build Prompt: ', t2 - t1)
        # print("--- LLM Prompt ---")
        # print(prompt)
        # print("------------------")

        # 3. Get the restocking decision from the LLM
        decision = self._get_llm_decision(prompt)
        if not decision:
            print("Failed to get a valid decision from the LLM. Aborting.")
            return

        t3 = time.time()
        print('Time to Get LLM Decision: ', t3 - t2)

        # 4. Process the decision and prepare data for update
        date_dt = datetime.strptime(date, "%Y-%m-%d")
        next_date = (date_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            restock_plan = decision['restock_plan']
            new_stock, new_balance = self._prepare_data_for_update(
                current_stock, restock_plan, current_balance, supplier_info, next_date
            )
        except (KeyError, ValueError, Exception) as e:
            print(f"Decision processing failed: {e}. Aborting.")
            return

        t4 = time.time()
        print('Time to Prepare update: ', t4 - t3)
        # 5. Update the DynamoDB tables in a transaction
        try:
            self.db_manager.update_state(
                old_stock_items=current_stock,
                new_stock_data=new_stock,
                old_balance_items=current_balance_items, 
                new_balance_data=new_balance
            )
            print(f"Restock cycle completed successfully. New balance: ${new_balance['balance']}")
        except Exception as e:
            print(f"Database update failed: {e}")
        
        t5 = time.time()
        print('Time to Upload to DB: ', t5 - t4)