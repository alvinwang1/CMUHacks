import os, glob, uuid, json, decimal, boto3
from datetime import datetime
from sentence_transformers import SentenceTransformer
from boto3.dynamodb.conditions import Key

ORG_ID = "demo"
TABLE = "EmbeddingsTable"
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table(TABLE)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk(text, max_words=400):
    out, buf = [], []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        if sum(len(x.split()) for x in buf) + len(para.split()) > max_words:
            out.append("\n\n".join(buf)); buf=[para]
        else:
            buf.append(para)
    if buf: out.append("\n\n".join(buf))
    return out

def to_decimal(vec):
    # DynamoDB Numbers are Decimals; keep 6 dp to shrink size
    return [decimal.Decimal(f"{x:.6f}") for x in vec]

def ingest_path(path, tier):
    uri = os.path.basename(path)
    text = open(path, encoding="utf-8").read()
    chs = chunk(text)
    for i, c in enumerate(chs):
        emb = model.encode([c])[0]
        item = {
            "PK": f"ORG#{ORG_ID}#GUIDELINES",
            "SK": f"DOC#{uri}#CHUNK#{i}",
            "content": c,
            "embedding": to_decimal(emb),
            "tier": tier,                # "SoT" or "Ref"
            "uri": uri,
            "span": f"L{i*100}-L{i*100+99}",
            "updated_at": datetime.utcnow().isoformat()+"Z",
        }
        table.put_item(Item=item)

# Example: mark *_policy.md as SoT, everything else Ref
for path in glob.glob("corpus/*"):
    tier = "SoT" if path.endswith("_policy.md") or "payment" in path or "identity" in path else "Ref"
    ingest_path(path, tier)

print("Ingested guideline chunks into DynamoDB.")
