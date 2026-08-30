import os
from dotenv import load_dotenv
from monday_api import fetch_board_metadata

load_dotenv()
wo_id = os.environ.get("MONDAY_WORK_ORDERS_BOARD_ID")
deals_id = os.environ.get("MONDAY_DEALS_BOARD_ID")

wo_meta = fetch_board_metadata(wo_id)
deals_meta = fetch_board_metadata(deals_id)

print("WORK ORDERS COLUMNS:")
for c in wo_meta['columns']:
    print(f"{c['id']}: {c['title']} ({c['type']})")

print("\nDEALS COLUMNS:")
for c in deals_meta['columns']:
    print(f"{c['id']}: {c['title']} ({c['type']})")
