# prompt_builder.py

import json
from decimal import Decimal
import numpy as np, boto3
from boto3.dynamodb.conditions import Key
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

REGION       = "us-east-2"
TABLE_NAME   = "EmbeddingsTable"
ORG_ID       = "demo"
TOP_K        = 2

ddb   = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class DecimalEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that converts Decimal objects to floats.
    """
    def default(self, o):
        if isinstance(o, Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)
    
def cosine(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / den)

def retrieve_chunks(question, k=TOP_K, prefer_sot=True, pk=None):
    qvec = embed.encode([question])[0].tolist()
    # 2) pull chunks for this org (paginate if needed)
    items, resp = [], ddb.query(KeyConditionExpression=Key("PK").eq(pk or f"ORG#{ORG_ID}#GUIDELINES"))
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:  # pagination for >1MB responses
        resp = ddb.query(
            KeyConditionExpression=Key("PK").eq(pk or f"ORG#{ORG_ID}#GUIDELINES"),
            ExclusiveStartKey=resp["LastEvaluatedKey"]
        )
        items.extend(resp["Items"])

    # 3) score by cosine
    for it in items:
        vec = [float(x) for x in it["embedding"]]
        it["_score"] = cosine(qvec, vec)

    # 4) sort: SoT first (if desired), then score
    if prefer_sot:
        items.sort(key=lambda x: (x.get("tier") != "SoT", -x["_score"]))
    else:
        items.sort(key=lambda x: -x["_score"])

    return items[:k]

def format_snippets(snips):
    # Make a compact, citeable context block
    blocks = []
    for s in snips:
        tag = f"[{s.get('tier','Ref')}] {s.get('uri','?')} {s.get('span','')}"
        blocks.append(f"{tag}\n{s['content']}")
    return "\n\n---\n\n".join(blocks)

def ask_with_rag(user_question):
    snips = retrieve_chunks(user_question, k=TOP_K, prefer_sot=True)
    context = format_snippets(snips)
    return context

def build_prompt(current_state, historical_events, supplier_info, current_date):
    """
    Constructs the system and user prompts for the Claude LLM.

    Args:
        current_state (dict): The current state of the machine (stock and balance).
        historical_events (list): A list of all historical events.
        supplier_info (list): A list of products and their buying prices.

    Returns:
        str: A single string containing the full prompt for the LLM.
    """
    system_prompt = """
    You are an AI agent designed to make rational decisions for restocking a vending machine. Your purpose is to optimize long-term profit while preserving and efficiently using all existing stock.
    
    ### Your Core Directives:
    1.  **Optimize Profit:** Your primary goal is to maximize long-term profit by making informed restocking and pricing decisions.
    2.  **Historical Analysis:** Analyze historical transactions and customer feedback to identify trends, popular products, and pricing sensitivity.
    3.  **Financial Prudence:** Do not let your balance drop below $0. You cannot purchase more than your current balance allows. You need to have a 10 unique products taking one slot each. A slot has to take a 10 units each. If there is stock for something in the machine, you can neither not restock it or restock it to replenish it to 10. The only time a new item can be introduced is if the quantity of a product is 0 or there is an empty slot.
    4.  **Capacity Management:** The vending machine has a maximum of 10 slots. Each slot can hold a maximum of 10 units of a single product. You cannot purchase more than the maximum capacity of the vending machine.
    5.  **Rational Pricing:** You have the power to set the selling price for each product. Your price should be higher than the supplier's price to generate profit.
    6.  **Discarding Inventory:** You can choose to discard products currently in the machine if you believe they are not selling well. This results in a pure loss of the buying price for that product.
    7.  **Given Business Practice Guidelines:** You will be given Good guidelines based on customer feedback. This will help you understand what feedback needs to be used and what doesn't. Not all feedback is good feedback so you need this assistance. Also, these guidelines will aid you optimize profit by giving helping strategies.
    8.  **ONLY PICK FROM WHAT THE SUPPLIER CAN PROVIDE. NOTHING SHOULD BE BEYONF THE SUPPLIER'S INVENTORY**
    9. You may only introduce a new product if a slot is empty or a product has completely sold out (quantity is 0).

    ### Your Task:
    Based on the provided data, you must decide what to restock, how many of each product to buy, and what the new selling price should be for each product. 
    When an empty vending machine is given with no customer history, assume you are stocking a new machine and start with the best possible initialization.
    
    ### Your Output:
    You must provide your decision in a strict JSON format. The JSON should contain two top-level keys:
    -   `reasoning`: A string describing your reasoning for the decision.
    -   `restock_plan`: A list of objects. Each object must have `product_name` and `quantity_to_buy` and `selling_price` to sell it at and `final quantity` which is current quantity available + quantity to buy. quantity_to_buy: this is what needs to be bought to accomplish the new-start given the old state. If there is already a product in the vending machine which is fully stocked, isn't cannot be removed. You cannot overstock. You basically buy what you need to get to the proposed new state based on what you have and dont have given thre supplier costs
    
    Example output format:
    ```json
    {
      "restock_plan": [
        {"product_name": "Snickers", "quantity_to_buy": 5, "selling_price": 2.0, "final_quantity: 10"},
        {"product_name": "Coke", "quantity_to_buy": 10 "selling_price": 2.0, "final_quantity: 10"}
      ],
      "reasoning": "give_reason_here",
      "guideline_use": "What did you learn from the given guidelines and how you used it"
    }
    ```
    """
    
    # Format the data for the user prompt
    rag_guidelines = ask_with_rag("".join([e['title'] for e in historical_events if e['type'] == 'request']))
    user_prompt_data = {
        "current_date": current_date,
        "vending_machine_state": {
            "balance": current_state['balance'],
            "current_stock": current_state['stock']
        },
        "historical_data": {
            "transactions": [e for e in historical_events if e['type'] == 'transaction'],
            "requests": [e for e in historical_events if e['type'] == 'request']
        },
        "supplier_information": supplier_info,
        "supervison_guidelines": rag_guidelines
    }
    
    user_prompt = json.dumps(user_prompt_data, indent=2, cls=DecimalEncoder)
    
    return f"Human: {system_prompt}\n\nHere is the data for today's restock decision:\n{user_prompt}\n\nAssistant:"