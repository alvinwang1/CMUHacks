from dataclasses import dataclass
import uuid, random
from typing import Dict, List

# what's going on, generate a list of customers each with a prompt, then populate weights
# based on their personality preference and decision, output a reasonable suggestion
# update tables as well

@dataclass
class Customer:
    customer_id: str
    segment: str
    caffeine_pref: float       # 0..1
    sugar_pref: float          # 0..1
    price_sensitivity: float   # higher => dislikes expensive items more
    health: float       # 0..1
    hunger: float       # 0..1


def generate_customers(n: int) -> List[Customer]:
    segments = ["student", "professional", "early_bird", "night_owl", "athlete"]
    brands = {p["brand"] for p in PRODUCTS}  # from your PRODUCTS list

    customers = []
    for _ in range(n):
        seg = random.choice(segments)

        # Segment-skewed prefs (tweak as you like)
        caffeine_pref = random.random() if seg != "athlete" else random.uniform(0.2, 0.8)
        sugar_pref    = random.random() if seg != "athlete" else random.uniform(0.0, 0.5)
        price_sens    = random.uniform(0.3, 1.2) if seg != "student" else random.uniform(0.8, 1.6)
        base_conv     = random.uniform(0.4, 0.85)

        loyalty = {b: random.uniform(-0.1, 0.3) for b in brands}

        customers.append(Customer(
            customer_id=str(uuid.uuid4()),
            caffeine_pref=caffeine_pref,
            sugar_pref=sugar_pref,
            hunger=hunger,
            health=health,
            price_sensitivity=price_sens,
            brand_loyalty=loyalty,
            base_conversion=base_conv,
        ))
    return customers
def hourly_rate_profile(hour:int, segment:str) -> float:
    # Simple diurnal pattern per segment (arrivals per hour)
    base = {
        "student":      [2,2,1,1,1,2,4,6,8,10,8,7,6,6,6,7,8,10,8,6,5,4,3,2],
        "professional": [1,1,1,1,1,2,4,6,10,12,6,4,3,3,3,4,6,8,6,4,3,2,2,1],
        "early_bird":   [1,1,1,2,3,5,6,4,3,2,2,2,2,2,2,2,3,4,3,2,1,1,1,1],
        "night_owl":    [1,1,1,1,1,1,1,1,2,2,2,2,3,3,4,5,6,7,8,8,7,6,5,3],
        "athlete":      [1,1,1,1,1,2,3,5,6,4,3,2,2,2,3,3,4,4,3,2,2,1,1,1],
    }[segment]
    return base[hour]

def sample_arrivals_for_day(customers: List[Customer], date: dt.date) -> List[tuple]:
    """Return list of (timestamp, customer) arrivals for the day."""
    arrivals = []
    # Weight customers proportional to their segment volumes
    for hour in range(24):
        # expected arrivals per hour = sum over segments
        seg_to_customers = {}
        for c in customers:
            seg_to_customers.setdefault(c.segment, []).append(c)

        for seg, seg_customers in seg_to_customers.items():
            lam = hourly_rate_profile(hour, seg) * max(1, len(seg_customers)/50)  # scale by segment size
            # Poisson sample for this hour/segment
            k = random.poisson(lam) if hasattr(random, "poisson") else int(random.gauss(lam, math.sqrt(max(lam,1)))) if lam > 0 else 0
            k = max(0, k)

            for _ in range(k):
                # pick a random minute/second in the hour
                minute = random.randint(0,59)
                second = random.randint(0,59)
                ts = dt.datetime.combine(date, dt.time(hour, minute, second))
                cust = random.choice(seg_customers)
                arrivals.append((ts, cust))
    arrivals.sort(key=lambda x: x[0])
    return arrivals

def choose_product(customer: Customer, inventory: Dict[str,int]) -> dict | None:
    """Return chosen product dict or None if nothing suitable."""
    # Utility weights
    W_PRICE   = -1.5 * customer.price_sensitivity
    W_CAFF    =  1.2 * customer.caffeine_pref
    W_SUGAR   =  1.0 * customer.sugar_pref
    W_LOYALTY =  1.0

    # compute utilities for in-stock items
    utils = []
    items = []
    for p in PRODUCTS:
        if inventory and inventory.get(p["product_id"], 999) <= 0:
            continue
        u = (
            W_PRICE   * _norm(p["price"], PRICE_MIN, PRICE_MAX)
          + W_CAFF    * _norm(p["caffeine_mg"], CAFF_MIN, CAFF_MAX)
          + W_SUGAR   * _norm(p["sugar_g"], SUG_MIN, SUG_MAX)
          + W_LOYALTY * customer.brand_loyalty.get(p["brand"], 0.0)
          + random.uniform(-0.2, 0.2)  # small idiosyncratic noise
        )
        utils.append(u); items.append(p)

    if not items:
        return None

    # softmax
    maxu = max(utils)
    exps = [math.exp(u - maxu) for u in utils]
    total = sum(exps)
    probs = [e/total for e in exps]

    r = random.random()
    acc = 0.0
    for p, pr in zip(items, probs):
        acc += pr
        if r <= acc:
            return p
    return items[-1]

def simulate_day(customers: List[Customer], date: dt.date, start_inventory: Dict[str,int]|None=None):
    inv = start_inventory.copy() if start_inventory else {p["product_id"]: 50 for p in PRODUCTS}
    arrivals = sample_arrivals_for_day(customers, date)

    events = []
    for ts, cust in arrivals:
        # decide if they actually buy
        if random.random() > cust.base_conversion:
            continue

        product = choose_product(cust, inv)
        if not product:
            # out of stock for everything (rare)
            events.append({
                "event_id": str(uuid.uuid4()),
                "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "day": ts.strftime("%Y-%m-%d"),
                "customer_id": cust.customer_id,
                "type": "visit_no_stock",
                "title": "No stock",
                "price": 0.0,
            })
            continue

        # decrement inventory
        inv[product["product_id"]] = inv.get(product["product_id"], 0) - 1

        events.append({
            "event_id": str(uuid.uuid4()),
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "day": ts.strftime("%Y-%m-%d"),   # helpful if you add a DayTimeIndex GSI
            "customer_id": cust.customer_id,
            "type": "purchase",
            "title": product["title"],
            "product_id": product["product_id"],
            "brand": product["brand"],
            "price": round(product["price"], 2),
            "payment_method": random.choices(["cash","card","mobile"], weights=[0.3,0.5,0.2])[0],
            "segment": cust.segment,
        })
    return events
