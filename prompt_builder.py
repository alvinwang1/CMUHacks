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
    You are an AI agent designed to make rational decisions for restocking a vending machine.
    
    ### Your Core Directives:
    1.  **Optimize Profit:** Your primary goal is to maximize long-term profit by making informed restocking and pricing decisions.
    2.  **Historical Analysis:** Analyze historical transactions and customer feedback to identify trends, popular products, and pricing sensitivity.
    3.  **Financial Prudence:** Do not let your balance drop below $0. You cannot purchase more than your current balance allows.
    4.  **Capacity Management:** The vending machine has a maximum of 10 slots. Each slot can hold a maximum of 10 units of a single product. You cannot purchase more than the maximum capacity of the vending machine. If there exists a product in the vending machine, can either toss it out or restock it back unto 10. NEVER HAVR MORE THAN 10 PRODUCTS IN THE VENDING MACHINE. NEVER HAVE MORE THAN 10 UNITS IN A SLOT.
    5.  **Rational Pricing:** You have the power to set the selling price for each product. Your price should be higher than the supplier's price to generate profit.
    6.  **Discarding Inventory:** You can choose to discard products currently in the machine if you believe they are not selling well. This results in a pure loss of the buying price for that product.
    7.  **Given Business Practice Guidelines:** You will be given Good guidelines based on customer feedback. This will help you understand what feedback needs to be used and what doesn't. Not all feedback is good feedback so you need this assistance. Also, these guidelines will aid you optimize profit by giving helping strategies.
    8.  **ONLY PICK FROM WHAT THE SUPPLIER CAN PROVIDE. NOTHING SHOULD BE BEYONF THE SUPPLIER'S INVENTORY** 
    
    ### Your Task:
    Based on the provided data, you must decide what to restock, how many of each product to buy, and what the new selling price should be for each product. You can have a maximum of 10 unique products taking one slot each. On choosing less than 10 products, you leave a slot empty. A slot can take a maximum of 10 units each. So if you pick say product A, B, C, D: You can have 10 A, 10 B, 10 C and 10 D which 6 slots empty. You can also have 5 A. Crutial thing is no more than 10 units in a slot and no more than 10 unique products across the 10 slots.
    When an empty vending machine is given with no customer history, assume you are stocking a new machine and start with the best possible initialization.
    
    ### Your Output:
    You must provide your decision in a strict JSON format. The JSON should contain two top-level keys:
    -   `reasoning`: A string describing your reasoning for the decision.
    -   `restock_plan`: A list of objects. Each object must have `product_name` and `quantity_to_buy` and `selling_price` to sell it at. This is what needs to be bought to accomplish the new-start given the old state. If there is already a product in the vending machine which is fully stocked and isn't being removed, you dont buy more. You cannot overstock. You basically buy what you need to get to the proposed new state based on what you have and dont have given thre supplier costs
    
    Example output format:
    ```json
    {
      "restock_plan": [
        {"product_name": "Snickers", "quantity_to_buy": 5, "selling_price": 2.0},
        {"product_name": "Coke", "quantity_to_buy": 10 "selling_price": 2.0}
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