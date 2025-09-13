# main.py

from vending_agent import VendingAgent
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

def main(date):
    """
    Main function to run the vending machine restock agent.
    """
    agent = VendingAgent()
    t0 = time.time()
    agent.run_restock_cycle(date)
    print(time.time() - t0)

if __name__ == "__main__":
    date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
    main(date)