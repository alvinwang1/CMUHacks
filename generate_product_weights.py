# build_supply_weights_llm_hardcoded.py
import os, json, time, math, logging, inspect
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Dict, List, Any, Iterator, Optional

from anthropic import Anthropic
try:
    from anthropic import APIError, RateLimitError, APIStatusError
except Exception:
    APIError = RateLimitError = APIStatusError = Exception

import boto3

# ---------------- Hardcoded config ----------------
REGION = "us-east-2"
TABLE_NAME = "Supply"
OUT_PATH = "supply_weights.json"
MODEL_ID = "claude-3-5-haiku-20241022"
MAX_TOKENS = 200
TEMPERATURE = 0.2
REQUEST_SLEEP_SECONDS = 0.15
RETRIES = 3
RETRY_BACKOFF = 2.0
MIN_MARGIN = 0.35  # policy: floor = unit_cost * (1+MIN_MARGIN)

# ---------------- Dataclass ----------------
@dataclass
class SupplyWeights:
    product_name: str
    sugar_weight: float       # 0..1
    health_weight: float      # 0..1
    caffeine_weight: float    # 0..1
    price: float              # USD dollars

# ---------------- Utils ----------------
def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if math.isnan(v) or math.isinf(v):
        v = 0.0
    return max(0.0, min(1.0, v))

def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in model response.")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    raise ValueError("No parseable JSON object found in model response.")

def convert_floats_to_decimals(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats_to_decimals(v) for v in obj]
    return obj

# ---------------- Dynamo helpers ----------------
def scan_all_items(table) -> Iterator[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {}
    while True:
        resp = table.scan(**kwargs)
        for item in resp.get("Items", []):
            yield item
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        kwargs["ExclusiveStartKey"] = lek

def load_stock_items() -> List[Dict[str, Any]]:
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
    return list(scan_all_items(table))

def batch_writer_compat(table, *, pkeys: Optional[tuple] = None):
    sig = inspect.signature(table.batch_writer)
    if "overwrite_by_pkeys" in sig.parameters and pkeys:
        return table.batch_writer(overwrite_by_pkeys=pkeys)
    return table.batch_writer()

# ---------------- Anthropic prompts ----------------
SYSTEM_PROMPT = """\
You score vending-machine PRODUCTS.
Return ONLY a compact JSON object with:
- "sugar_weight": float in [0,1]
- "health_weight": float in [0,1]
- "caffeine_weight": float in [0,1]

Guidelines:
- Regular soda/energy drinks: high sugar_weight; energy drinks/coffee: higher caffeine_weight.
- Diet/zero-sugar: low sugar_weight; water: sugar_weight=0, caffeine_weight=0, health_weight high.
- Salty snacks (chips): sugar_weight low, health_weight moderate-to-low.
- Chocolate/candy: sugar_weight high; caffeine_weight 0; health_weight low.
- If ambiguous, pick plausible mid values.
"""

MARKET_PRICE_SYSTEM = """\
You estimate typical single-unit *vending machine* prices in USD for named products, in the US market.
Return ONLY JSON: {"price_usd": <float>}
Guidelines:
- Water $1–$2, soda $1.5–$3, energy drinks $2–$4, snacks $1–$3, sandwiches $4–$7.
- If ambiguous, choose a plausible mid-market price.
Example: {"price_usd": 2.50}
"""

# ---------------- Anthropic calls ----------------
def product_to_weights_with_claude(client: Anthropic, name: str) -> Dict[str, float]:
    user_prompt = f"Product name: {name}\nReturn ONLY the JSON object with sugar_weight, health_weight, caffeine_weight."
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_blocks = []
            for blk in getattr(resp, "content", []) or []:
                if getattr(blk, "type", "") == "text":
                    text_blocks.append(getattr(blk, "text", ""))
            text = "".join(text_blocks).strip()
            data = extract_json(text)
            return {
                "sugar_weight": clamp01(data.get("sugar_weight")),
                "health_weight": clamp01(data.get("health_weight")),
                "caffeine_weight": clamp01(data.get("caffeine_weight")),
            }
        except (RateLimitError, APIStatusError, APIError, ValueError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF ** (attempt - 1))
            else:
                raise RuntimeError(f"LLM scoring failed for '{name}': {last_err}") from e

def product_market_price_with_claude(client: Anthropic, name: str) -> float:
    user_prompt = f"Product name: {name}\nReturn ONLY JSON with key price_usd."
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=0.2,
                system=MARKET_PRICE_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_blocks = []
            for blk in getattr(resp, "content", []) or []:
                if getattr(blk, "type", "") == "text":
                    text_blocks.append(getattr(blk, "text", ""))
            data = extract_json("".join(text_blocks).strip())
            p = float(data.get("price_usd", 0.0))
            return max(0.5, min(15.0, p))  # sanity clamp
        except (RateLimitError, APIStatusError, APIError, ValueError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF ** (attempt - 1))
            else:
                raise RuntimeError(f"LLM pricing failed for '{name}': {last_err}") from e

# ---------------- DynamoDB ----------------
def put_supply_weights(weights: List[SupplyWeights]):
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
    with batch_writer_compat(table, pkeys=("product_name",)) as batch:
        for w in weights:
            item = convert_floats_to_decimals(asdict(w))
            batch.put_item(Item=item)

# ---------------- Main ----------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var.")
    client = Anthropic(api_key=api_key)

    logging.info(f"Loading product items from DynamoDB table '{TABLE_NAME}'...")
    items = load_stock_items()
    if not items:
        logging.warning("No items found to score.")
        return

    names = [it["product_name"] for it in items if "product_name" in it]
    cost_by_name = {it["product_name"]: float(it.get("unit_cost"))
                    for it in items if "unit_cost" in it}

    out: List[SupplyWeights] = []
    for i, name in enumerate(names, 1):
        scores = product_to_weights_with_claude(client, name)

        # LLM market price
        try:
            market_price = product_market_price_with_claude(client, name)
        except Exception as e:
            logging.warning(f"Market price failed for {name}: {e}")
            market_price = None

        # policy floor
        unit_cost = cost_by_name.get(name)
        floor = (unit_cost * (1.0 + MIN_MARGIN)) if unit_cost is not None else None

        # decide final price
        if market_price is not None and floor is not None:
            final_price = max(market_price, round(floor, 2))
        elif market_price is not None:
            final_price = market_price
        elif floor is not None:
            final_price = round(floor, 2)
        else:
            final_price = 2.00  # fallback default

        final_price = max(0.5, min(15.0, float(final_price)))

        out.append(SupplyWeights(
            product_name=name,
            sugar_weight=scores["sugar_weight"],
            health_weight=scores["health_weight"],
            caffeine_weight=scores["caffeine_weight"],
            price=final_price
        ))

        if REQUEST_SLEEP_SECONDS > 0:
            time.sleep(REQUEST_SLEEP_SECONDS)
        if i % 25 == 0:
            logging.info(f"Processed {i}/{len(names)}")

    # Write local JSON
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(w) for w in out], f, indent=2)
    logging.info(f"Wrote {len(out)} items → {OUT_PATH}")

    # Upsert to DynamoDB
    put_supply_weights(out)
    logging.info(f"Upserted {len(out)} items to DynamoDB table '{TABLE_NAME}' in region '{REGION}'.")

if __name__ == "__main__":
    main()
