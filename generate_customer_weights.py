import os
import re
import json
import time
import math
import argparse
import logging
from datetime import datetime, timedelta
from copy import deepcopy
import random
import uuid
from decimal import Decimal
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple

# --- Anthropic imports (robust) ---
from anthropic import Anthropic
try:
    from anthropic import APIError, RateLimitError, APIStatusError
except Exception:  # fallback if versions differ
    APIError = RateLimitError = APIStatusError = Exception

import boto3
from boto3.dynamodb.conditions import Attr

region = "us-east-2"
table_name = "Customers"

MODEL_ID = "claude-3-5-haiku-20241022"
MAX_OUTPUT_TOKENS = 250
TEMPERATURE = 0.4
REQUEST_SLEEP_SECONDS = 0.15
RETRIES = 3
RETRY_BACKOFF = 2.0

@dataclass
class Customer:
    customer_id: str
    segment: str
    caffeine_pref: float       # 0..1
    sugar_pref: float          # 0..1
    price_sensitivity: float   # 0..1 (higher => dislikes expensive items more)
    health: float              # 0..1
    hunger: float              # 0..1

def clamp01(x: Any) -> float:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    if math.isnan(f) or math.isinf(f):
        f = 0.0
    return max(0.0, min(1.0, f))

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first parseable JSON object from text.
    Uses a balanced-brace scan (since Python 're' doesn't support (?R)).
    """
    text = text.strip()
    # Quick path
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    in_str = False
    esc = False
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
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    raise ValueError("No parseable JSON object found in model response.")

def load_personalities(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("personalities.json must be an object: {segment: description, ...}")
    return data

def to_customer(segment: str, payload: Dict[str, Any]) -> Customer:
    return Customer(
        customer_id=str(uuid.uuid4()),
        segment=segment,
        caffeine_pref=clamp01(payload.get("caffeine_pref")),
        sugar_pref=clamp01(payload.get("sugar_pref")),
        price_sensitivity=clamp01(payload.get("price_sensitivity")),
        health=clamp01(payload.get("health")),
        hunger=clamp01(payload.get("hunger")),
    )

SYSTEM_PROMPT = """\
You convert a vending-machine customer persona into numeric traits in [0,1].

Return ONLY a compact JSON object (no prose, no code fences) with keys:
- "caffeine_pref": float in [0,1]
- "sugar_pref": float in [0,1]
- "price_sensitivity": float in [0,1]  (higher => more cost-averse)
- "health": float in [0,1]             (health-consciousness)
- "hunger": float in [0,1]             (tendency to seek satiating items)

Guidelines:
- Map language about caffeine/energy/coffee/night shifts → higher caffeine_pref.
- Candy/soda/sweets/treats → higher sugar_pref; water/protein/diet/healthy → lower.
- Budget/tight/coins/fixed income → higher price_sensitivity; expensed/“barely notice cost” → lower.
- Health-conscious/gym/coach/protein/water → higher health; “indulge/cheat/candy/chips” → lower.
- Meals/filling/stay full/salty/snack-as-meal/skip meals → higher hunger.
- If ambiguous, choose a reasonable mid value (≈0.4–0.6).

Output EXAMPLE:
{"caffeine_pref":0.62,"sugar_pref":0.48,"price_sensitivity":0.71,"health":0.33,"hunger":0.58}
"""

def persona_to_traits_with_claude(
    client: Anthropic,
    model: str,
    name: str,
    description: str,
    temperature: float = 0.2,
    max_output_tokens: int = 200,
    retries: int = 3,
    backoff: float = 2.0,
) -> Dict[str, Any]:
    user_prompt = (
        f"Persona name: {name}\nPersona description:\n{description}\n\n"
        "Return ONLY the JSON object."
    )

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_out = "".join(
                getattr(blk, "text", "")
                for blk in getattr(resp, "content", [])
                if getattr(blk, "type", "") == "text"
            )
            data = extract_json(text_out)
            return {
                "caffeine_pref": clamp01(data.get("caffeine_pref")),
                "sugar_pref": clamp01(data.get("sugar_pref")),
                "price_sensitivity": clamp01(data.get("price_sensitivity")),
                "health": clamp01(data.get("health")),
                "hunger": clamp01(data.get("hunger")),
            }
        except (RateLimitError, APIStatusError, APIError, ValueError) as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(backoff ** (attempt - 1))

    raise RuntimeError(f"Anthropic mapping failed for persona '{name}': {last_err}")

def build_customers_with_llm(
    personalities_path: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    sleep_between_reqs: float,
) -> List[Customer]:
    personas = load_personalities(personalities_path)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY in your environment.")

    client = Anthropic(api_key=api_key)

    customers: List[Customer] = []
    for i, (segment, desc) in enumerate(personas.items(), start=1):
        traits = persona_to_traits_with_claude(
            client=client,
            model=model,
            name=segment,
            description=desc,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        customers.append(to_customer(segment, traits))
        if sleep_between_reqs > 0:
            time.sleep(sleep_between_reqs)
    return customers

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Change this path if needed
    personalities_path = "corpus/personalities.json"

    customers = build_customers_with_llm(
        personalities_path=personalities_path,
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        sleep_between_reqs=REQUEST_SLEEP_SECONDS,
    )

    # Pretty-print JSON
    print(json.dumps([asdict(c) for c in customers], indent=2))
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    from decimal import Decimal

    def convert_floats_to_decimals(obj):
        """
        Recursively converts float values in a dictionary or list to Decimal objects.
        This is necessary for DynamoDB, which does not support float types directly.
        """
        if isinstance(obj, float):
            # Convert float to string first to avoid potential precision issues
            # when converting directly from float to Decimal.
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats_to_decimals(elem) for elem in obj]
        else:
            return obj

    for c in customers:
        item_to_put = convert_floats_to_decimals(asdict(c))
        table.put_item(Item=item_to_put)
    print("Customers written to DynamoDB.")


if __name__ == "__main__":
    main()
