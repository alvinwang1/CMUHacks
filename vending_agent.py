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
        """Calculates the total cost of the restock plan.
        Accepts restock_plan either as:
          - dict: { product_name: { 'quantity_to_buy': X, ... }, ... }
          - list: [ { 'product_name': name, 'quantity_to_buy': X, ... }, ... ]
        """
        # build supplier price map (use Decimal if available)
        supplier_prices = {}
        for item in supplier_info:
            name = item.get('product_name')
            price = item.get('price')
            # normalize numeric type
            try:
                supplier_prices[name] = float(price)
            except (TypeError, ValueError):
                supplier_prices[name] = 0.0

        total_cost = 0.0

        # handle dict or list shapes for restock_plan
        if isinstance(restock_plan, dict):
            iterator = restock_plan.items()  # (product_name, details)
            for product_name, details in iterator:
                qty = 0
                if isinstance(details, dict):
                    qty = details.get('quantity_to_buy', 0)
                else:
                    # fallback if details is a primitive
                    try:
                        qty = int(details)
                    except Exception:
                        qty = 0
                price = supplier_prices.get(product_name)
                if price is None:
                    print(f"Warning: supplier price missing for '{product_name}'. Skipping cost.")
                    continue
                try:
                    total_cost += float(qty) * float(price)
                except Exception:
                    print(f"Warning: cost calc failed for '{product_name}'. Skipping.")
        elif isinstance(restock_plan, list):
            for item in restock_plan:
                product_name = item.get('product_name')
                qty = item.get('quantity_to_buy', 0)
                price = supplier_prices.get(product_name)
                if price is None:
                    print(f"Warning: supplier price missing for '{product_name}'. Skipping cost.")
                    continue
                try:
                    total_cost += float(qty) * float(price)
                except Exception:
                    print(f"Warning: cost calc failed for '{product_name}'. Skipping.")
        else:
            print("Warning: unknown restock_plan type for cost calculation. Assuming cost 0.")
        return total_cost


    def _prepare_data_for_update(self, old_stock, restock_plan, old_balance, supplier_info, new_date):
        """
        Prepares the data for the database update transaction based on the LLM's restock plan.
        Returns (new_stock_data, new_balance_data).
        """
        # Convert old stock list to a dict
        old_stock_dict = {item['product_name']: int(item.get('quantity', 0)) for item in old_stock}

        new_stock_data = []
        restocked_products = set()

        # collect product names from both old stock and restock plan (support both shapes)
        product_names = set(old_stock_dict.keys())
        if isinstance(restock_plan, dict):
            product_names.update(restock_plan.keys())
        elif isinstance(restock_plan, list):
            for item in restock_plan:
                name = item.get('product_name')
                if name:
                    product_names.add(name)

        # helper to get details for a product from restock_plan
        def _get_plan_details(product_name):
            if isinstance(restock_plan, dict):
                return restock_plan.get(product_name, {})
            elif isinstance(restock_plan, list):
                for it in restock_plan:
                    if it.get('product_name') == product_name:
                        return it
            return {}

        for product_name in product_names:
            restocked_products.add(product_name)
            details = _get_plan_details(product_name)

            # quantity_to_buy may be missing (treat as 0)
            try:
                qty_to_buy = int(details.get('quantity_to_buy', 0))
            except (ValueError, TypeError, AttributeError):
                qty_to_buy = 0
                print(f"Warning: Quantity for '{product_name}' is not a valid number. Setting to 0.")

            # selling_price may be missing (keep previous or skip if none and no old price)
            selling_price_val = details.get('selling_price', None)
            selling_price = None
            if selling_price_val is not None:
                try:
                    selling_price = Decimal(str(selling_price_val))
                except (ValueError, TypeError, InvalidOperation):
                    selling_price = None
                    print(f"Warning: Selling price for '{product_name}' is not a valid number. Skipping item.")

            current_qty = old_stock_dict.get(product_name, 0)
            new_qty = current_qty + qty_to_buy

            # enforce capacity = 10 gracefully
            if new_qty > 10:
                print(f"Warning: New qty for '{product_name}' ({new_qty}) exceeds capacity 10. Capping to 10.")
                new_qty = 10

            # only add items with positive stock
            if new_qty > 0:
                entry = {
                    'stock_id': str(uuid.uuid4()),
                    'product_name': product_name,
                    'quantity': new_qty,
                    'is_actual': 1,
                    'date': new_date,
                    'time_of_day': 'opening'
                }
                if selling_price is not None:
                    entry['selling_price'] = selling_price
                new_stock_data.append(entry)

        # Calculate total cost using the robust cost function
        total_cost = self._calculate_restock_cost(restock_plan, supplier_info)

        # normalize old_balance -> numeric
        try:
            current_balance_val = float(old_balance)
        except Exception:
            try:
                # if old_balance is a dict or list, attempt to extract
                current_balance_val = float(old_balance[0].get('balance', 0))
            except Exception:
                current_balance_val = float(self.initial_budget)

        new_balance = current_balance_val - float(total_cost)

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