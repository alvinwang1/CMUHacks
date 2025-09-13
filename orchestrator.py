from datetime import datetime, timedelta
from vending_sim_customer_day import day_sim
from dotenv import load_dotenv
from vending_agent import VendingAgent
from delete_tables import delete_all_tables

load_dotenv()


def sim_night_and_next_day(date, current_balance):
    agent = VendingAgent(init_budget=current_balance)
    print('==================> Stocking Up <==================')
    agent.run_restock_cycle(date=date)
    print('==================> Stocked Up <==================')
    date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    day_sim(date)
    print('==================> Day Simulated <==================')


if __name__ == "__main__":
    delete_all_tables()

    balance = input("Enter starting balance: ")
    days = input('How many days: ')
    
    date = datetime.now().strftime("%Y-%m-%d")
    for day in range(int(days)):
        date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
        sim_night_and_next_day(date, balance)