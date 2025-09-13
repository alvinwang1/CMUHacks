# pip install boto3 sentence-transformers anthropic numpy
import os, math, numpy as np, boto3
from boto3.dynamodb.conditions import Key, Attr
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

REGION       = "us-east-2"
TABLE_NAME   = "EmbeddingsTable"
ORG_ID       = "demo"
TOP_K        = 8

ddb   = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
anth  = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

SYSTEM = """You are an advisor for a vending shop manager.  
- Base ALL decisions strictly on Source-of-Truth (SoT) guideline snippets.  
- If no SoT snippet supports an action, refuse and ask for the missing policy/data.  
- Do not describe taking actions yourself; only advise.  
- Respond clearly and briefly.  
- End every response with a one-line “Recommended Action:” that summarizes the single best next step (e.g., adjust a price, restock an item, contact maintenance). """

def ask_with_rag(user_question: str):
    snips = retrieve_chunks(user_question, k=TOP_K, prefer_sot=True)
    context = format_snippets(snips)

    prompt = [
        {"type":"text","text": f"Question:\n{user_question}"},
        {"type":"text","text": "Guideline snippets (use citations like [SoT] file Lx-Ly):\n" + context}
    ]

    resp = anth.messages.create(
        model="claude-3-7-sonnet-20250219",   # use your available Claude model
        system=SYSTEM,
        max_tokens=600,
        messages=[{"role":"user","content": prompt}]
    )

    # Return model text + the citations we used
    answer = "".join([p.text for p in resp.content if getattr(p, "text", None)])
    # attach the top-k metadata for UI/tooling
    cites = [{"uri": s["uri"], "span": s["span"], "tier": s.get("tier","Ref"), "score": round(s["_score"], 3)} for s in snips]
    return answer, cites

# query the tables
table_name = "events"
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(table_name)
# grab records from table
def get_records():
    day = "2025-01-01"
    start_key = None
    items = []
    while True:
        kwargs = {
            "FilterExpression": Attr("time").begins_with(day)  # "YYYY-MM-DD HH:MM:SS"
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break
    result = ""
    for i in range(len(items)):
        result += items[i]["title"] + " "
    return result
if __name__ == "__main__":
    # test out prompting LLM
    # pulling query from feedback, which would be in the database
    prompt = get_records()
    ans, cites = ask_with_rag(prompt)
    print(ans, "\n\nCitations:", cites)


