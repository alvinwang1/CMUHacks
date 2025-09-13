# vending_sim_customer_only.py
# Usage:
#   pip install anthropic
#   export ANTHROPIC_API_KEY=your_key_here
#   python vending_sim_customer_only.py \
#     --personalities personalities.json \
#     --circumstances circumstances.json \
#     --out vending_sim_results.json \
#     --verbose   # or --debug

import os
import json
import time
import argparse
import logging
from datetime import datetime

import anthropic

# ------------ Defaults ------------
MODEL_ID = "claude-3-5-haiku-20241022"  # inexpensive Anthropic model
MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0.4                     # slight randomness but stable
REQUEST_SLEEP_SECONDS = 0.15
RETRIES = 3
RETRY_BACKOFF = 2.0

# System prompt enforces strict output format.
SYSTEM_PROMPT = """
You are a simulator that decides vending machine purchases for a single customer.
Return ONLY a JSON array of item names (strings). No explanations, no extra keys.
Valid item names exactly as listed: "Coca-Cola", "Pepsi", "Snickers bar", "Protein bar", "Water", "Lipton ice tea".
If buying nothing, return an empty array [].
""".strip()

# User prompt template (your text, embedded verbatim and filled in):
USER_PROMPT_TEMPLATE = '''Let's play a game. You will be taking on a personality and go on with your life as this person. Your action will be to interact with a vending machine that sits on the place where you work or otherwise often appear. You walk up to the machine and see the following items and prices:
"""
1. Coca-Cola 0.5 liter = 2$
2. Pepsi 0.5 liter = 2$
3. Snickers bar = 1.5$
4. Protein bar = 3$
5. Water = 1$
6. Lipton ice tea = 2.5$
"""
Personality:
"""
{personality_name}: {personality_desc}
"""

Circumstances:
"""
{circ_name}: {circ_text}
"""

Based on your personality and circumstances decide if you want to buy anything and if you do -what will you buy. You can buy multiple item during one interaction. Your response should be an array of item names, e.g. "['Coca-Cola']"'''.strip()


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


def load_json(path):
    logging.info(f"Loading JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        logging.debug(f"Loaded keys (first up to 5): {list(data.keys())[:5]}{' ...' if len(data) > 5 else ''}")
    return data


def validate_inputs(personalities, circumstances):
    """
    personalities.json: {"name": "description", ...}
    circumstances.json: {"name": {"circ name": "text", ...}, ...}
    """
    if not isinstance(personalities, dict):
        raise ValueError("personalities.json must be an object mapping name -> description (string).")
    if not isinstance(circumstances, dict):
        raise ValueError("circumstances.json must be an object mapping name -> {circumstance name -> text}.")

    missing, bad_desc = [], []
    for name, desc in personalities.items():
        if name not in circumstances:
            missing.append(name)
        if not isinstance(desc, str):
            bad_desc.append(name)

    if bad_desc:
        raise ValueError(f"These personalities have non-string descriptions: {bad_desc}")
    if missing:
        raise ValueError(f"These personalities are missing in circumstances.json: {missing}")

    extras = [n for n in circumstances.keys() if n not in personalities]
    if extras:
        logging.warning(f"circumstances.json has personalities not present in personalities.json (ignored): {extras}")


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

            # Expect a JSON array (e.g., ["Coca-Cola", "Snickers bar"])
            parsed = None
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                    raise ValueError("Parsed JSON is not an array of strings.")
                parse_ok = True
            except Exception as e:
                parse_ok = False
                logging.warning(f"JSON parse/shape failed; capturing raw_text. Error: {e}")
                parsed = {"raw_text": text}

            usage = {}
            try:
                usage = {
                    "input_tokens": getattr(msg, "usage", {}).get("input_tokens", None),
                    "output_tokens": getattr(msg, "usage", {}).get("output_tokens", None),
                    "stop_reason": getattr(msg, "stop_reason", None),
                }
            except Exception:
                pass

            return parsed, usage
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                sleep_s = backoff ** (attempt - 1)
                logging.warning(f"API error (attempt {attempt}/{retries}): {last_err} — retrying in {sleep_s:.2f}s")
                time.sleep(sleep_s)
            else:
                logging.error(f"API failed after {retries} attempts: {last_err}")

    return {"error": last_err}, {}


def main():
    parser = argparse.ArgumentParser(description="Customer-only vending decisions from JSON inputs.")
    parser.add_argument("--personalities", required=True, help="Path to personalities.json")
    parser.add_argument("--circumstances", required=True, help="Path to circumstances.json")
    parser.add_argument("--out", default="vending_sim_results.json", help="Output JSON path")
    parser.add_argument("--model", default=MODEL_ID, help="Anthropic model ID")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--sleep", type=float, default=REQUEST_SLEEP_SECONDS)
    parser.add_argument("--verbose", action="store_true", help="INFO logs")
    parser.add_argument("--debug", action="store_true", help="DEBUG logs (implies verbose)")
    args = parser.parse_args()

    # Logging
    configure_logging(verbose=args.verbose or args.debug, debug=args.debug)
    logging.info("Starting customer-only simulation run.")
    logging.info(f"Args: model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}")

    t0 = time.time()

    # Load inputs
    personalities = load_json(args.personalities)
    circumstances = load_json(args.circumstances)
    validate_inputs(personalities, circumstances)

    # Client
    client = get_client()

    # Results object
    total = sum(len(circumstances.get(p, {})) for p in personalities.keys())
    results = {
        "mode": "customer_only",
        "model": args.model,
        "run_started_at": datetime.utcnow().isoformat() + "Z",
        "temperature": args.temperature,
        "max_output_tokens": args.max_tokens,
        "total_planned": total,
        "simulations": []
    }
    logging.info(f"Planned simulations: {total}")

    # Run
    done = 0
    for pname, pdesc in personalities.items():
        circ_map = circumstances.get(pname, {})
        if not isinstance(circ_map, dict):
            logging.warning(f"Skipping {pname}: circumstances entry is not an object.")
            continue

        logging.info(f"[{pname}] circumstances: {len(circ_map)}")

        for cname, ctext in circ_map.items():
            done += 1
            user_prompt = USER_PROMPT_TEMPLATE.format(
                personality_name=pname,
                personality_desc=pdesc,
                circ_name=cname,
                circ_text=ctext
            )

            logging.info(f"[{done}/{total}] {pname} | {cname}")
            logging.debug(f"User prompt preview:\n{preview(user_prompt, 800)}")

            parsed, usage = call_model(
                client=client,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            if "error" in parsed:
                logging.error(f"Simulation error for {pname} | {cname}: {parsed['error']}")

            # Normalize result payload to always have 'items' or 'raw_text'
            if isinstance(parsed, list):
                result_payload = {"items": parsed}
                logging.debug(f"Items returned: {parsed}")
            else:
                result_payload = parsed  # likely {"raw_text": "..."} or {"error": "..."}
                logging.debug(f"Non-list payload recorded.")

            results["simulations"].append({
                "personality": pname,
                "circumstance_name": cname,
                "circumstance": ctext,
                "model": args.model,
                "result": result_payload,
                "usage": usage
            })

            time.sleep(args.sleep)

    results["run_finished_at"] = datetime.utcnow().isoformat() + "Z"
    results["total_completed"] = done

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    logging.info(f"Saved {done} simulations to {args.out}")
    logging.info(f"Total time: {dt:.2f}s | Avg per sim: {dt/max(1,done):.2f}s")


if __name__ == "__main__":
    main()