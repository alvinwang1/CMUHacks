# vending_sim_customer_day_dot_db_hardcoded.py
# LLM-free vending day simulator with hardcoded config for your exact DynamoDB schema.

import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import random

import boto3
from boto3.dynamodb.conditions import Attr

# ----------------------- Hardcoded config -----------------------
AWS_REGION       = "us-east-2"

TABLE_CUSTOMERS  = "Customers"   # customer_id, sugar_pref, health, caffeine_pref, hunger, price_sensitivity, segment
TABLE_SUPPLY     = "Supply"      # product_name, sugar_weight, health_weight, caffeine_weight, price
TABLE_STOCK      = "stock"       # product_name, quantity, price, stock_id, is_actual
TABLE_EVENTS     = "events"      # event_id (PK)
TABLE_BALANCE    = "balance"     # trans_id (PK), is_active

OUT_PATH         = "vending_sim_results.json"

# Scoring & behavior
PRICE_ALPHA_SCALE = 1.0          # scale for price_sensitivity -> price penalty
THRESHOLD_BASE    = 0.0          # base threshold; buy if best_score >= threshold
HUNGER_BONUS      = 0.25         # threshold -= HUNGER_BONUS * hunger
EPSILON           = 0.00         # exploration prob (choose 2nd-best)

# Run pacing / order
EVENT_STEP_SECS   = 90           # seconds between events
SLEEP_BETWEEN     = 0.00         # no API calls
SHUFFLE_CUSTOMERS = False
RAND_SEED         = 42           # set to None for nondeterministic

# Simulation clock
SIM_DATE          = datetime.utcnow().date().isoformat()
START_TIME        = "09:00:00"

LOG_LEVEL         = logging.INFO

# ----------------------- Logging -----------------------
def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------- Helpers -----------------------
def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v != v or v in (float("inf"), float("-inf")):
        v = 0.0
    return max(0.0, min(1.0, v))

def as_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return default

def batch_writer_compat(table, *, pkeys: Optional[tuple] = None):
    import inspect
    sig = inspect.signature(table.batch_writer)
    if "overwrite_by_pkeys" in sig.parameters and pkeys:
        return table.batch_writer(overwrite_by_pkeys=pkeys)
    return table.batch_writer()

def build_assortment_block(stock_state: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(sorted(stock_state.keys()), start=1):
        m = stock_state[item]
        lines.append(f"{idx}. {item} = ${m.get('price', 0)} (remaining: {m.get('quantity', 0)})")
    return "\n".join(lines)

# ----------------------- DynamoDB I/O -----------------------
def get_tables():
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return (
        dynamodb.Table(TABLE_CUSTOMERS),
        dynamodb.Table(TABLE_SUPPLY),
        dynamodb.Table(TABLE_STOCK),
        dynamodb.Table(TABLE_EVENTS),
        dynamodb.Table(TABLE_BALANCE),
    )

def scan_all(table, **kwargs):
    resp = table.scan(**kwargs)
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **kwargs)
        items.extend(resp.get("Items", []))
    return items

def load_customers_from_db(customers_table) -> Dict[str, Dict[str, float]]:
    """
    { customer_id: {sugar, health, caffeine, hunger, price_sensitivity} } (all clamped to [0,1])
    """
    items = scan_all(customers_table)
    out: Dict[str, Dict[str, float]] = {}
    for it in items:
        cid = it.get("customer_id")
        if not cid:
            continue
        out[cid] = {
            "sugar":            clamp01(it.get("sugar_pref", 0.0)),
            "health":           clamp01(it.get("health", 0.0)),
            "caffeine":         clamp01(it.get("caffeine_pref", 0.0)),
            "hunger":           clamp01(it.get("hunger", 0.0)),
            "price_sensitivity":clamp01(it.get("price_sensitivity", 0.0)),
            # "segment": it.get("segment")  # available if you want to segment analytics later
        }
    if not out:
        logging.warning("No customers loaded from Customers.")
    else:
        logging.info(f"Loaded {len(out)} customers.")
    return out

def load_supply_weights_from_db(supply_table) -> Dict[str, Dict[str, float]]:
    """
    { product_name: {sugar_weight, health_weight, caffeine_weight, price? (float)} }
    """
    items = scan_all(supply_table)
    out: Dict[str, Dict[str, float]] = {}
    for it in items:
        name = it.get("product_name")
        if not name:
            continue
        out[name] = {
            "sugar_weight":    clamp01(it.get("sugar_weight", 0.0)),
            "health_weight":   clamp01(it.get("health_weight", 0.0)),
            "caffeine_weight": clamp01(it.get("caffeine_weight", 0.0)),
            "price":           as_float(it.get("price", 0.0), 0.0),
        }
    if not out:
        logging.warning("No product weights loaded from Supply.")
    else:
        logging.info(f"Loaded weights for {len(out)} products.")
    return out

def load_stock_from_db(stock_table, supply_prices: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Prefer price from stock actuals; fall back to Supply price if stock price missing.
    Returns { product_name: {"quantity": int, "price": float, "stock_id": str} }
    """
    stock: Dict[str, Dict[str, Any]] = {}
    items = scan_all(stock_table, FilterExpression=Attr("is_actual").eq(1))
    for it in items:
        name = it.get("product_name")
        if not name:
            continue
        qty = int(as_float(it.get("quantity", 0), 0.0))
        price_val = it.get("price", None)
        if price_val is None:
            price = float(supply_prices.get(name, 0.0))
        else:
            price = as_float(price_val, float(supply_prices.get(name, 0.0)))
        stock[name] = {"quantity": qty, "price": price, "stock_id": it.get("stock_id")}
    if not stock:
        logging.warning("Stock is empty (no is_actual==true rows).")
    else:
        logging.info("Loaded current stock (is_actual=1):")
        for nm in sorted(stock.keys()):
            m = stock[nm]
            logging.info(f"- {nm}: qty={m['quantity']}, price=${m['price']}")
    return stock

def batch_write_events(events_table, events: List[Dict[str, Any]]):
    if not events:
        return
    with batch_writer_compat(events_table, pkeys=("event_id",)) as batch:
        for e in events:
            price = e.get("price", 0)
            if not isinstance(price, Decimal):
                e["price"] = Decimal(str(price))
            batch.put_item(Item=e)

def update_stock_actuals(stock_table, final_stock_state: Dict[str, Dict[str, Any]], sim_date: str, time_of_day: str = "closing"):
    # Mark existing actuals as non-actual
    items = scan_all(stock_table, FilterExpression=Attr("is_actual").eq(1))
    for it in items:
        stock_id = it.get("stock_id")
        if not stock_id:
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

    # Insert fresh actual snapshot
    with stock_table.batch_writer() as batch:
        for name, meta in sorted(final_stock_state.items()):
            qty = int(meta.get("quantity", 0))
            price = float(meta.get("price", 0.0))
            sid = meta.get("stock_id") or str(uuid.uuid4())
            item = {
                "stock_id": sid,
                "product_name": name,
                "quantity": qty,
                "price": Decimal(str(price)),
                "is_actual": 1,
                "date": sim_date,
                "time_of_day": time_of_day,
            }
            batch.put_item(Item=item)

def read_active_balance(balance_table) -> Decimal:
    try:
        items = scan_all(balance_table, FilterExpression=Attr("is_active").eq(1))
        if not items:
            return Decimal("0")
        items.sort(key=lambda i: (i.get("date", ""), i.get("time_of_day", "")), reverse=True)
        val = items[0].get("balance", Decimal("0"))
        if isinstance(val, Decimal): return val
        if isinstance(val, (int, float)): return Decimal(str(val))
        if isinstance(val, str): return Decimal(val)
        return Decimal("0")
    except Exception as e:
        logging.warning(f"Failed to read active balance: {e}")
        return Decimal("0")

def deactivate_active_balances(balance_table):
    try:
        items = scan_all(balance_table, FilterExpression=Attr("is_active").eq(1))
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

# ----------------------- Scoring & Purchase -----------------------
def pick_item_for_customer(
    cw: Dict[str, float],
    stock_state: Dict[str, Dict[str, Any]],
    pw_by_item: Dict[str, Dict[str, float]],
    *,
    epsilon: float,
    rng: random.Random,
) -> Tuple[Optional[str], float]:
    """
    score = cw·pw - alpha * normalized_price
    alpha = PRICE_ALPHA_SCALE * cw['price_sensitivity']
    threshold = THRESHOLD_BASE - HUNGER_BONUS * cw['hunger']

    Returns (best_item_or_None, best_score).
    If best_score < threshold -> returns (None, best_score) so caller buys nothing.
    """
    alpha = PRICE_ALPHA_SCALE * clamp01(cw.get("price_sensitivity", 0.0))
    threshold = THRESHOLD_BASE - HUNGER_BONUS * clamp01(cw.get("hunger", 0.0))

    # normalize price among in-stock items
    prices = [float(m["price"]) for m in stock_state.values() if m.get("quantity", 0) > 0]
    pmin, pmax = (min(prices), max(prices)) if prices else (0.0, 1.0)
    def norm_price(p: float) -> float:
        if pmax == pmin:
            return 0.0
        return (p - pmin) / (pmax - pmin)

    # score only in-stock items that have weights
    scored: List[Tuple[str, float]] = []
    for item, meta in stock_state.items():
        if meta.get("quantity", 0) <= 0:
            continue
        w = pw_by_item.get(item)
        if not w:
            continue
        score = (
            cw["sugar"]    * w["sugar_weight"] +
            cw["health"]   * w["health_weight"] +
            cw["caffeine"] * w["caffeine_weight"]
            - alpha * norm_price(float(meta["price"]))
        )
        scored.append((item, score))

    if not scored:
        return None, float("-inf")

    # choose best (optionally explore second-best)
    scored.sort(key=lambda t: t[1], reverse=True)
    best_item, best_score = scored[0]

    # allow opting out
    if best_score < threshold:
        return None, best_score

    if epsilon > 0.0 and len(scored) > 1 and rng.random() < epsilon:
        return scored[1][0], scored[1][1]

    return best_item, best_score

def fulfill_purchase(requested_items, stock_state: Dict[str, Dict[str, Any]]):
    if not isinstance(requested_items, list):
        return [], requested_items, 0.0
    fulfilled, rejected = [], []
    total_spent = 0.0
    for name in requested_items:
        if fulfilled:
            rejected.append(name); continue
        meta = stock_state.get(name)
        if not meta or meta.get("quantity", 0) <= 0:
            rejected.append(name); continue
        fulfilled.append(name)
        meta["quantity"] -= 1
        total_spent += float(meta.get("price", 0.0))
    return fulfilled, rejected, round(total_spent, 2)

# ----------------------- Orchestration -----------------------
def day_sim(date):
    print(f"[{date}] Simulating all day...")
    configure_logging()
    rng = random.Random()

    customers_table, supply_table, stock_table, events_table, balance_table = get_tables()

    customers = load_customers_from_db(customers_table)          # {cid: {...}}
    supply = load_supply_weights_from_db(supply_table)           # {product_name: {...}}
    supply_price_map = {k: v.get("price", 0.0) for k, v in supply.items()}

    stock_initial = load_stock_from_db(stock_table, supply_price_map)
    for title, meta in stock_initial.items():
        if not isinstance(meta.get("price"), (int, float)):
            meta["price"] = 0.0

    sequence: List[str] = list(customers.keys())
    if SHUFFLE_CUSTOMERS:
        rng.shuffle(sequence)

    stock_state = deepcopy(stock_initial)
    total_spend = 0.0
    events_to_write: List[Dict[str, Any]] = []

    try:
        start_dt = datetime.strptime(f"{date} {START_TIME}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        start_dt = datetime.strptime(f"{date} 09:00:00", "%Y-%m-%d %H:%M:%S")
    next_event_dt = start_dt

    interactions: List[Dict[str, Any]] = []

    for i, cid in enumerate(sequence, 1):
        c = customers[cid]
        best_item, score = pick_item_for_customer(
            cw=c,
            stock_state=stock_state,
            pw_by_item={k: {"sugar_weight": v["sugar_weight"],
                            "health_weight": v["health_weight"],
                            "caffeine_weight": v["caffeine_weight"]} for k, v in supply.items()},
            epsilon=EPSILON,
            rng=rng,
        )
        if best_item is None:
            thr = THRESHOLD_BASE - HUNGER_BONUS * clamp01(c.get("hunger", 0.0))
            logging.info(f"[{i}/{len(sequence)}] {cid} -> no purchase (best_score={score:.3f} < threshold={thr:.3f})")
        chosen = [best_item] if best_item else []
        fulfilled, rejected, spend = fulfill_purchase(chosen, stock_state)
        total_spend += spend

        if fulfilled:
            title = fulfilled[0]
            price_now = stock_state.get(title, {}).get("price", 0.0)
            events_to_write.append({
                "event_id": str(uuid.uuid4()),
                "price": Decimal(str(price_now)),
                "time": next_event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "type": "transaction",
                "customer": cid,
            })
            next_event_dt += timedelta(seconds=EVENT_STEP_SECS)
        else:
            title = "None"
            price_now = stock_state.get(title, {}).get("price", 0.0)
            events_to_write.append({
                "event_id": str(uuid.uuid4()),
                "price": Decimal(str(price_now)),
                "time": next_event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "type": "transaction",
                "customer": cid,
            })
            next_event_dt += timedelta(seconds=EVENT_STEP_SECS)

        logging.info(f"[{i}/{len(sequence)}] {cid} -> pick={best_item} score={score:.3f} "
                     f"fulfilled={fulfilled} spend=${spend:.2f}")

        interactions.append({
            "customer": cid,
            "picked_item": best_item,
            "score": None if score == float("-inf") else round(score, 6),
            "fulfilled_items": fulfilled,
            "rejected_items": rejected,
            "amount_spent": spend,
            "hunger": c.get("hunger"),
            "price_sensitivity": c.get("price_sensitivity"),
        })

        if SLEEP_BETWEEN > 0:
            import time as _t; _t.sleep(SLEEP_BETWEEN)

    results = {
        "mode": "customer_only_day_dot_db_hardcoded",
        "run_started_at": datetime.utcnow().isoformat() + "Z",
        "run_finished_at": datetime.utcnow().isoformat() + "Z",
        "total_planned_interactions": len(sequence),
        "total_completed_interactions": len(sequence),
        "total_amount_spent": round(total_spend, 2),
        "params": {
            "PRICE_ALPHA_SCALE": PRICE_ALPHA_SCALE,
            "THRESHOLD_BASE": THRESHOLD_BASE,
            "HUNGER_BONUS": HUNGER_BONUS,
            "EPSILON": EPSILON,
        },
        "starting_stock": stock_initial,
        "ending_stock": stock_state,
        "interactions": interactions,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved {len(sequence)} interactions to {OUT_PATH}")
    logging.info(f"Total spent across the day: ${results['total_amount_spent']:.2f}")

    # Persist to DynamoDB
    logging.info(f"Writing {len(events_to_write)} events to '{TABLE_EVENTS}'.")
    batch_write_events(events_table, events_to_write)

    logging.info("Updating stock actuals in DynamoDB.")
    update_stock_actuals(stock_table, stock_state, sim_date=SIM_DATE, time_of_day="closing")

    logging.info("Updating balance in DynamoDB.")
    prev_balance = read_active_balance(balance_table)
    new_balance = prev_balance + Decimal(str(results["total_amount_spent"]))
    deactivate_active_balances(balance_table)
    write_new_balance(balance_table, sim_date=SIM_DATE, time_of_day="closing", new_balance=new_balance)
    logging.info(f"Balance updated: previous={prev_balance} closing={new_balance}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
