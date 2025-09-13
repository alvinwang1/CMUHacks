# vending_sim_customer_day.py
# Simulate a "day" of customers buying from a vending machine with finite stock.
#
# Usage:
#   pip install anthropic
#   export ANTHROPIC_API_KEY=your_key_here
#   python vending_sim_customer_day.py \
#     --personalities personalities.json \
#     --circumstances circumstances.json \
#     --stock stock.json \
#     --out vending_sim_results.json \
#     --verbose    # or --debug
#
# personalities.json: {"name": "description", ...}
# circumstances.json: {"name": {"circ_name": "text", ...}, ...}
# stock.json: {"Item Name": {"quantity": N, "price": Y}, ...}

import os
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from copy import deepcopy
import random
import uuid
from decimal import Decimal
from typing import Optional

import anthropic
import boto3
from boto3.dynamodb.conditions import Attr

# ------------ Defaults ------------
MODEL_ID = "claude-3-5-haiku-20241022"  # default to latest/cheap Claude 3.5 Haiku
MAX_OUTPUT_TOKENS = 250
TEMPERATURE = 0.4
REQUEST_SLEEP_SECONDS = 0.15
RETRIES = 3
RETRY_BACKOFF = 2.0

# System prompt: force strict array of strings from CURRENT assortment only
SYSTEM_PROMPT = """
You decide vending-machine purchases for a single customer.
Return EITHER:
- a JSON array of item names (strings), OR
- a JSON object that may include optional keys: "items" (array of item names) and "request" (string with a short wish for something missing).
No prose, no extra keys, no comments.
- Choose from the CURRENT assortment exactly as shown in the user message.
- You may buy AT MOST ONE item. If buying, the array (when present) must contain exactly one item name. If buying nothing, use an empty array or omit the "items" field.
- Do NOT invent items. Do NOT select items with zero remaining quantity.
""".strip()

# User prompt template: fills in the live assortment + personality + circumstance
USER_PROMPT_TEMPLATE = '''Let's play a game. You will be taking on a personality and go on with your life as this person. Your action will be to interact with a vending machine that sits where you work or otherwise often appear.

You walk up to the machine and see the following items, prices, and remaining quantities:

{assortment_block}

Personality:
"""
{personality_name}: {personality_desc}
"""

Circumstances:
"""
{circ_name}: {circ_text}
"""

Based on your personality and circumstances decide if you want to buy anything and if you do — what will you buy.
If the content of the machine doesn't fully satisfy you you can leave a request for the it's operator. It's possible to buy something but still leave a request for new items.
Respond with JSON of the following format:
{{"items": ["Coca-Cola"], "request": "Please add sparkling water."}}
If you have no request, omit the "request" field. If you haven't bought anything then omit the 'items' list
'''.strip()


def configure_logging(verbose: bool, debug: bool):
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_json(path, expect_obj=True):
    logging.info(f"Loading JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if expect_obj and not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object at the top level.")
    logging.debug(f"Loaded keys (first 5): {list(data.keys())[:5]}{' ...' if len(data) > 5 else ''}")
    return data


def validate_inputs(personalities, circumstances, stock):
    # personalities: dict[name] -> str
    # circumstances: dict[name] -> dict[circ_name] -> str
    # stock: dict[item] -> {"quantity": int, "price": number}
    if not isinstance(personalities, dict):
        raise ValueError("personalities.json must be an object mapping name -> description (string).")
    if not isinstance(circumstances, dict):
        raise ValueError("circumstances.json must be an object mapping name -> {circ_name -> text}.")
    if not isinstance(stock, dict):
        raise ValueError("stock.json must be an object mapping item -> {quantity, price}.")

    # personalities vs circumstances
    missing = [p for p in personalities if p not in circumstances]
    if missing:
        raise ValueError(f"These personalities are missing in circumstances.json: {missing}")

    # stock sanity
    for item, meta in stock.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Stock entry for '{item}' must be an object with quantity, price.")
        if "quantity" not in meta or "price" not in meta:
            raise ValueError(f"Stock entry for '{item}' missing 'quantity' or 'price'.")
        if not isinstance(meta["quantity"], int) or meta["quantity"] < 0:
            raise ValueError(f"Stock quantity for '{item}' must be a non-negative integer.")
        if not (isinstance(meta["price"], int) or isinstance(meta["price"], float)):
            raise ValueError(f"Stock price for '{item}' must be numeric.")


def build_assortment_block(stock_state):
    # Show only items with quantity >= 0 (we'll show 0 to discourage selection)
    # Format: "1) Item = $Y (remaining: N)"
    # Keep consistent ordering by name
    lines = []
    for idx, item in enumerate(sorted(stock_state.keys()), start=1):
        meta = stock_state[item]
        qty = meta.get("quantity", 0)
        price = meta.get("price", 0)
        lines.append(f"{idx}. {item} = ${price} (remaining: {qty})")
    return "\n".join(lines)


# ---------------- DynamoDB helpers ----------------
def get_tables(region: str, events_table_name: str, stock_table_name: str, balance_table_name: str):
    dynamodb = boto3.resource("dynamodb", region_name=region)
    return (
        dynamodb.Table(events_table_name),
        dynamodb.Table(stock_table_name),
        dynamodb.Table(balance_table_name),
    )


def load_prices_from_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If file is a full stock.json, convert to {title: price}
        prices = {}
        if isinstance(data, dict) and data and isinstance(next(iter(data.values())), dict):
            for title, meta in data.items():
                p = meta.get("price")
                if isinstance(p, (int, float)):
                    prices[title] = float(p)
        elif isinstance(data, dict):
            # Already a mapping of title -> price
            for title, p in data.items():
                if isinstance(p, (int, float)):
                    prices[title] = float(p)
        return prices
    except Exception as e:
        logging.warning(f"Failed to load prices from {path}: {e} Falling back to zero prices.")
        return {}


def load_stock_from_db(stock_table, prices_map: dict) -> dict:
    """Read current actual stock (is_actual == 1) and return mapping:
    { product_name: {"quantity": int, "price": float} }
    Prefers price from DB if present; falls back to prices_map.
    Also prints the loaded stock for visibility.
    """
    stock = {}
    scan_kwargs = {"FilterExpression": Attr("is_actual").eq(1)}
    resp = stock_table.scan(**scan_kwargs)
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = stock_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
        items.extend(resp.get("Items", []))

    for it in items:
        name = it.get("product_name") or it.get("title")
        qty = it.get("quantity", 0)
        if name is None:
            continue
        try:
            qty = int(qty)
        except Exception:
            qty = 0
        # Prefer DB price, fall back to provided prices map
        db_price = it.get("price")
        if isinstance(db_price, Decimal):
            price = float(db_price)
        elif isinstance(db_price, (int, float)):
            price = float(db_price)
        elif isinstance(db_price, str):
            try:
                price = float(db_price)
            except Exception:
                price = None
        else:
            price = None
        if price is None:
            price = float(prices_map.get(name, 0.0))
        stock[name] = {"quantity": qty, "price": price}

    if not stock:
        logging.warning("Stock table returned no actual items (is_actual=1). Starting with empty stock.")
    else:
        logging.info("Loaded current stock (is_actual=1):")
        for name in sorted(stock.keys()):
            meta = stock[name]
            logging.info(f"- {name}: qty={meta['quantity']}, price=${meta['price']}")
    return stock


def batch_write_events(events_table, events: list[dict]):
    if not events:
        return
    with events_table.batch_writer(overwrite_by_pkeys=("event_id",)) as batch:
        for e in events:
            # Convert price to Decimal
            price = e.get("price", 0)
            if not isinstance(price, Decimal):
                e["price"] = Decimal(str(price))
            batch.put_item(Item=e)


def update_stock_actuals(stock_table, final_stock_state: dict, sim_date: str, time_of_day: str = "closing"):
    # 1) Mark previous actuals as non-actual
    scan_kwargs = {"FilterExpression": Attr("is_actual").eq(1)}
    resp = stock_table.scan(**scan_kwargs)
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = stock_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
        items.extend(resp.get("Items", []))

    for it in items:
        stock_id = it.get("stock_id")
        if not stock_id:
            # Cannot update without key; skip with warning
            logging.warning("Encountered stock item without stock_id; skipping actual reset.")
            continue
        try:
            stock_table.update_item(
                Key={"stock_id": stock_id},
                UpdateExpression="SET is_actual = :zero",
                ExpressionAttributeValues={":zero": 0},
            )
        except Exception as e:
            logging.warning(f"Failed to update stock actual flag for {stock_id}: {e}")

    # 2) Insert new actual snapshot rows
    with stock_table.batch_writer() as batch:
        for name, meta in sorted(final_stock_state.items()):
            qty = int(meta.get("quantity", 0))
            price = meta.get("price", 0.0)
            stock_id = f"{sim_date}#{time_of_day}#{name}"
            item = {
                "stock_id": stock_id,
                "product_name": name,
                "quantity": qty,
                "price": Decimal(str(price)),
                "is_actual": 1,
                "date": sim_date,
                "time_of_day": time_of_day,
            }
            batch.put_item(Item=item)


def read_active_balance(balance_table) -> Decimal:
    """Read the currently active balance row(s) (is_active == 1) and return the numeric balance.
    If multiple active rows exist, pick the max by (date, time_of_day) to be deterministic.
    """
    try:
        resp = balance_table.scan(FilterExpression=Attr("is_active").eq(1))
        items = resp.get("Items", [])
        while "LastEvaluatedKey" in resp:
            resp = balance_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], FilterExpression=Attr("is_active").eq(1))
            items.extend(resp.get("Items", []))
        if not items:
            return Decimal("0")
        def sort_key(i):
            return (i.get("date", ""), i.get("time_of_day", ""))
        items.sort(key=sort_key, reverse=True)
        val = items[0].get("balance", Decimal("0"))
        if isinstance(val, Decimal):
            return val
        if isinstance(val, (int, float)):
            return Decimal(str(val))
        if isinstance(val, str):
            return Decimal(val)
        return Decimal("0")
    except Exception as e:
        logging.warning(f"Failed to read active balance: {e}")
        return Decimal("0")


def deactivate_active_balances(balance_table):
    try:
        resp = balance_table.scan(FilterExpression=Attr("is_active").eq(1))
        items = resp.get("Items", [])
        while "LastEvaluatedKey" in resp:
            resp = balance_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], FilterExpression=Attr("is_active").eq(1))
            items.extend(resp.get("Items", []))
        for it in items:
            trans_id = it.get("trans_id")
            if not trans_id:
                logging.warning("Balance row missing trans_id; skipping deactivate.")
                continue
            try:
                balance_table.update_item(
                    Key={"trans_id": trans_id},
                    UpdateExpression="SET is_active = :zero",
                    ExpressionAttributeValues={":zero": 0},
                )
            except Exception as e:
                logging.warning(f"Failed to deactivate balance row {trans_id}: {e}")
    except Exception as e:
        logging.warning(f"Failed to scan active balances: {e}")


def write_new_balance(balance_table, sim_date: str, time_of_day: str, new_balance: Decimal):
    try:
        item = {
            "trans_id": str(uuid.uuid4()),
            "date": sim_date,
            "time_of_day": time_of_day,
            "is_active": 1,
            "balance": new_balance if isinstance(new_balance, Decimal) else Decimal(str(new_balance)),
        }
        balance_table.put_item(Item=item)
    except Exception as e:
        logging.warning(f"Failed to write new balance: {e}")


def get_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logging.error("ANTHROPIC_API_KEY not set.")
    else:
        logging.debug("ANTHROPIC_API_KEY detected.")
    return anthropic.Anthropic()


def preview(text: str, n: int = 200) -> str:
    if text is None:
        return ""
    return text if len(text) <= n else text[:n] + " ..."


def call_model(client, model, system_prompt, user_prompt,
               max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE,
               retries=RETRIES, backoff=RETRY_BACKOFF):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            logging.debug(f"API attempt {attempt}: model={model}, temp={temperature}, max_tokens={max_tokens}")
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = ""
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    text += block.text

            logging.debug(f"Raw response preview: {preview(text)}")

            # Expect either:
            #   1) a JSON array of strings (items), or
            #   2) a JSON object that may include optional "items" (array) and/or "request" (string)
            try:
                parsed_json = json.loads(text)
                if isinstance(parsed_json, list):
                    if not all(isinstance(x, str) for x in parsed_json):
                        raise ValueError("Array is not all strings.")
                    normalized = {"items": parsed_json, "request": None}
                elif isinstance(parsed_json, dict):
                    items = parsed_json.get("items", None)
                    if items is not None:
                        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                            raise ValueError("Object 'items' must be a list of strings if present.")
                    req = parsed_json.get("request", None)
                    if req is not None and not isinstance(req, str):
                        raise ValueError("'request' must be a string if present.")
                    normalized = {"items": items, "request": req}
                else:
                    raise ValueError("Top-level JSON must be array or object.")
                return normalized, extract_usage(msg)
            except Exception as e:
                logging.warning(f"JSON parse/shape failed; capturing raw_text. Error: {e}")
                return {"raw_text": text}, extract_usage(msg)

        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                sleep_s = backoff ** (attempt - 1)
                logging.warning(f"API error (attempt {attempt}/{retries}): {last_err} — retrying in {sleep_s:.2f}s")
                time.sleep(sleep_s)
            else:
                logging.error(f"API failed after {retries} attempts: {last_err}")
                return {"error": last_err}, {}
    # unreachable
    return {"error": last_err}, {}


def extract_usage(msg):
    usage = {}
    try:
        usage = {
            "input_tokens": getattr(msg, "usage", {}).get("input_tokens", None),
            "output_tokens": getattr(msg, "usage", {}).get("output_tokens", None),
            "stop_reason": getattr(msg, "stop_reason", None),
        }
    except Exception:
        pass
    return usage


def fulfill_purchase(requested_items, stock_state):
    """
    requested_items: list[str] or {"raw_text": "..."} or {"error": "..."}
    stock_state: dict[item] -> {quantity, price}
    Returns:
      fulfilled_items: list[str] (0 or 1 item actually dispensed)
      rejected_items: list[str] (rejected due to invalid item or out-of-stock)
      total_spent: float
    Side-effects:
      Decrements stock_state quantities for fulfilled items.
    """
    if not isinstance(requested_items, list):
        # Not a valid list — nothing purchased
        return [], requested_items, 0.0

    fulfilled = []
    rejected = []
    total_spent = 0.0

    # Only accept at most one item per person
    for name in requested_items:
        if fulfilled:
            # Per-person limit reached; reject any additional items
            rejected.append(name)
            continue
        item_meta = stock_state.get(name)
        if not item_meta:
            rejected.append(name)
            continue
        if item_meta["quantity"] <= 0:
            rejected.append(name)
            continue
        # Accept this single unit (limit 1 per person)
        fulfilled.append(name)
        item_meta["quantity"] -= 1
        total_spent += float(item_meta["price"])

    return fulfilled, rejected, round(total_spent, 2)


def _run_customer_day(
    personalities_path: str,
    circumstances_path: str,
    *,
    aws_region: str = "us-east-2",
    events_table_name: str = "events",
    stock_table_name: str = "stock",
    balance_table_name: str = "balance",
    model: str = MODEL_ID,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    request_sleep_seconds: float = REQUEST_SLEEP_SECONDS,
    shuffle: bool = False,
    seed: Optional[int] = None,
    date: Optional[str] = None,
    start_time: str = "09:00:00",
    event_step_secs: int = 90,
    prices_path: Optional[str] = None,
    out_path: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
):
    """Core engine for simulating a customer-only day.
    Returns the results dict. Also writes events/stock/balance to DynamoDB.
    """
    configure_logging(verbose=verbose or debug, debug=debug)
    logging.info("Starting day simulation.")
    logging.info(f"Args: model={model}, temp={temperature}, max_tokens={max_tokens}")

    personalities = load_json(personalities_path)
    circumstances = load_json(circumstances_path)

    # Price source: optional prices file; otherwise rely on DB prices or 0
    prices_map = load_prices_from_json(prices_path)

    # DynamoDB tables
    events_table, stock_table, balance_table = get_tables(
        region=aws_region,
        events_table_name=events_table_name,
        stock_table_name=stock_table_name,
        balance_table_name=balance_table_name,
    )

    # Load stock from DB (actuals)
    stock_initial = load_stock_from_db(stock_table, prices_map)

    if not stock_initial:
        logging.warning("Stock is empty; simulation will proceed but no purchases can be fulfilled.")

    # Validate price types for what we have.
    for title, meta in stock_initial.items():
        if not isinstance(meta.get("price"), (int, float)):
            meta["price"] = 0.0

    rng = random.Random(seed) if seed is not None else random

    # Copy stock to a mutable state for the day
    stock_state = deepcopy(stock_initial)

    # Build a sequence of (pname, pdesc, cname, ctext) with one random circumstance per personality
    sequence = []
    for pname, pdesc in personalities.items():
        circ_map = circumstances.get(pname, {})
        if isinstance(circ_map, dict) and circ_map:
            cname, ctext = rng.choice(list(circ_map.items()))
            sequence.append((pname, pdesc, cname, ctext))
        else:
            logging.warning(f"No circumstances found for {pname}; skipping.")

    if shuffle:
        rng.shuffle(sequence)

    client = get_client()

    results = {
        "mode": "customer_only_day",
        "model": model,
        "run_started_at": datetime.utcnow().isoformat() + "Z",
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "total_planned_interactions": len(sequence),
        "interactions": [],
        "starting_stock": stock_initial,  # snapshot of inputs
    }

    total_spend = 0.0
    done = 0
    events_to_write = []

    # Prepare consistent timestamps across the simulated day
    sim_date = date or datetime.utcnow().date().isoformat()
    try:
        start_dt = datetime.strptime(f"{sim_date} {start_time}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        start_dt = datetime.strptime(f"{sim_date} 09:00:00", "%Y-%m-%d %H:%M:%S")
    next_event_dt = start_dt

    for pname, pdesc, cname, ctext in sequence:
        done += 1
        # Prepare current assortment block from live stock
        assortment_block = build_assortment_block(stock_state)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            assortment_block=assortment_block,
            personality_name=pname,
            personality_desc=pdesc,
            circ_name=cname,
            circ_text=ctext,
        )

        logging.info(f"[{done}/{len(sequence)}] {pname} | {cname}")
        logging.debug(f"Prompt preview:\n{preview(user_prompt, 1200)}")

        # Call model
        parsed, usage = call_model(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Determine items and optional request (both optional)
        model_items = None
        model_request = None
        if isinstance(parsed, dict) and ("items" in parsed or "request" in parsed or "raw_text" in parsed or "error" in parsed):
            if isinstance(parsed.get("items"), list):
                model_items = parsed.get("items")
            if isinstance(parsed.get("request"), str):
                model_request = parsed.get("request")
        elif isinstance(parsed, list):  # backward compatibility
            model_items = parsed
        # else: leave both as None

        # Fulfill with current stock
        fulfilled, rejected, spend = fulfill_purchase(model_items, stock_state)
        total_spend += spend

        if isinstance(parsed, list):
            logging.debug(f"Requested items: {parsed}")
        else:
            logging.debug(f"Non-list model output recorded.")

        if rejected:
            logging.info(f"Rejected due to invalid/out-of-stock: {rejected}")
        if fulfilled:
            logging.info(f"Fulfilled: {fulfilled} | Spent: ${spend:.2f}")
        else:
            logging.info("No items fulfilled.")
        if model_request:
            logging.info(f"Customer request: {model_request}")

        # Create events with sequential timestamps
        # Purchase event (at most one item per person)
        if fulfilled:
            title = fulfilled[0]
            events_to_write.append({
                "event_id": str(uuid.uuid4()),
                "price": Decimal(str(stock_state.get(title, {}).get("price", 0.0))),
                "time": next_event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "type": "transaction",
            })
            next_event_dt += timedelta(seconds=event_step_secs)
        # Request event
        if model_request:
            events_to_write.append({
                "event_id": str(uuid.uuid4()),
                "price": Decimal("0"),
                "time": next_event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": model_request,
                "type": "request",
            })
            next_event_dt += timedelta(seconds=event_step_secs)

        results["interactions"].append({
            "personality": pname,
            "circumstance_name": cname,
            "circumstance": ctext,
            "assortment_snapshot": assortment_block,
            "model_items": model_items,                # list or null
            "model_request": model_request,            # string or null
            "model_raw_text": parsed.get("raw_text") if isinstance(parsed, dict) else None,
            "fulfilled_items": fulfilled,
            "rejected_items": rejected,  # invalid or exceeded stock
            "amount_spent": spend,
            "usage": usage
        })

        time.sleep(request_sleep_seconds)

    results["run_finished_at"] = datetime.utcnow().isoformat() + "Z"
    results["total_completed_interactions"] = done
    results["total_amount_spent"] = round(total_spend, 2)
    results["ending_stock"] = stock_state  # what’s left after the day

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {done} interactions to {out_path}")
    logging.info(f"Total spent across the day: ${results['total_amount_spent']:.2f}")
    # ---------------- Write to DynamoDB ----------------
    logging.info(f"Writing {len(events_to_write)} events to DynamoDB table '{events_table_name}'.")
    batch_write_events(events_table, events_to_write)

    logging.info("Updating stock actuals in DynamoDB.")
    update_stock_actuals(stock_table, stock_state, sim_date=sim_date, time_of_day="closing")

    logging.info("Updating balance in DynamoDB.")
    prev_balance = read_active_balance(balance_table)
    logging.info(f"Starting balance (active): {prev_balance}")
    new_balance = prev_balance + Decimal(str(results["total_amount_spent"]))
    # Deactivate any currently active rows, then write the closing snapshot
    deactivate_active_balances(balance_table)
    write_new_balance(balance_table, sim_date=sim_date, time_of_day="closing", new_balance=new_balance)
    logging.info(f"Balance updated: previous={prev_balance} closing={new_balance}")
    logging.info("Done.")

    return results


def simulate_vending_day(personalities_path: str, circumstances_path: str):
    """Public entrypoint with only mandatory inputs.
    - personalities_path: path to personalities.json
    - circumstances_path: path to circumstances.json
    Uses sensible defaults for everything else and returns the results dict.
    """
    return _run_customer_day(
        personalities_path=personalities_path,
        circumstances_path=circumstances_path,
    )


def main():
    # Keep CLI compatibility; delegate to the core engine.
    parser = argparse.ArgumentParser(description="Simulate a day of customer-only vending decisions with live stock and persist to DynamoDB.")
    parser.add_argument("--personalities", required=True, help="Path to personalities.json")
    parser.add_argument("--circumstances", required=True, help="Path to circumstances.json")
    parser.add_argument("--stock", required=False, help="Optional path to stock.json for price fallback {item: {quantity, price}}")
    parser.add_argument("--prices", required=False, help="Optional path to prices JSON mapping {title: price}. If omitted, will try --stock for prices.")
    parser.add_argument("--out", default="vending_sim_results.json", help="Output JSON path")
    parser.add_argument("--model", default=MODEL_ID, help="Anthropic model ID (default: claude-3-5-haiku-20241022)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--sleep", type=float, default=REQUEST_SLEEP_SECONDS)
    parser.add_argument("--shuffle", action="store_true", help="Randomize the order of customers/circumstances")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for picking one circumstance per personality and optional shuffling")
    parser.add_argument("--verbose", action="store_true", help="INFO logs")
    parser.add_argument("--debug", action="store_true", help="DEBUG logs (implies verbose)")
    # DynamoDB config
    parser.add_argument("--aws_region", default="us-east-2", help="AWS region for DynamoDB (default from test.py: us-east-2)")
    parser.add_argument("--events_table", default="events", help="DynamoDB table name for events")
    parser.add_argument("--stock_table", default="stock", help="DynamoDB table name for stock")
    parser.add_argument("--balance_table", default="balance", help="DynamoDB table name for balance")
    # Simulation day/time config
    parser.add_argument("--date", default=datetime.utcnow().date().isoformat(), help="Simulation date (YYYY-MM-DD). Default: today (UTC)")
    parser.add_argument("--start_time", default="09:00:00", help="Start time for the first event (HH:MM:SS)")
    parser.add_argument("--event_step_secs", type=int, default=90, help="Seconds between successive events (default: 90)")
    args = parser.parse_args()

    _run_customer_day(
        personalities_path=args.personalities,
        circumstances_path=args.circumstances,
        aws_region=args.aws_region,
        events_table_name=args.events_table,
        stock_table_name=args.stock_table,
        balance_table_name=args.balance_table,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_sleep_seconds=args.sleep,
        shuffle=args.shuffle,
        seed=args.seed,
        date=args.date,
        start_time=args.start_time,
        event_step_secs=args.event_step_secs,
        prices_path=args.prices or args.stock,
        out_path=args.out,
        verbose=args.verbose,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
