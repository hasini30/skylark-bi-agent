import asyncio
from agent import BIAgent
from monday_api import list_boards, fetch_board_metadata, fetch_board_items
import board_review as br
import interpret
from dotenv import load_dotenv
import os

load_dotenv()

def run_test():
    print("Fetching boards...")
    board_ids = [os.environ["MONDAY_WORK_ORDERS_BOARD_ID"], os.environ["MONDAY_DEALS_BOARD_ID"]]
    pulled = []
    for bid in board_ids:
        meta = fetch_board_metadata(bid)
        items = fetch_board_items(bid)
        table_name = "work_orders" if bid == board_ids[0] else "deals"
        pulled.append((table_name, None, bid, meta, items))
        
    print("Reviewing boards...")
    reviews_and_frames = []
    for p in pulled:
        table, label, bid, meta, items = p
        reviews_and_frames.append(br.review_board(bid, meta, items, table, label=label, interpret_with=interpret.read_board, merge_reading=interpret.merge))
        
    reviews = [r for r, f in reviews_and_frames]
    frames = {p[0]: f for p, (r, f) in zip(pulled, reviews_and_frames)}
    
    workspace = br.compare_all(reviews, frames)
    db_conn = br.build_database(frames, workspace.all_findings)
    
    agent = BIAgent(db_conn, workspace)
    
    questions = [
        "total pipeline",
        "work-order revenue",
        "sector pipeline"
    ]
    
    for q in questions:
        print(f"\n--- Q: {q} ---")
        agent.messages = [{"role": "system", "content": agent._system_prompt()}]
        result = agent.send_message(q)
        print(result)

run_test()
