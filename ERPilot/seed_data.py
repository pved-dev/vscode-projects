import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

URL = os.getenv("ERPNEXT_URL")
API_KEY = os.getenv("ERPNEXT_API_KEY")
API_SECRET = os.getenv("ERPNEXT_API_SECRET")

headers = {
    "Authorization": f"token {API_KEY}:{API_SECRET}",
    "Content-Type": "application/json"
}

def create(doctype, data):
    res = requests.post(
        f"{URL}/api/resource/{doctype}",
        headers=headers,
        json=data
    )
    if res.status_code in [200, 201]:
        print(f"✅ Created {doctype}: {data.get('name') or data.get('item_name') or data.get('supplier_name') or ''}")
    else:
        print(f"❌ Failed {doctype}: {res.text[:100]}")
    return res

# 1. Item Groups
create("Item Group", {"item_group_name": "Raw Materials", "parent_item_group": "All Item Groups"})
create("Item Group", {"item_group_name": "Components", "parent_item_group": "All Item Groups"})

# 2. Items
items = [
    {"item_code": "RM-STEEL-001", "item_name": "Steel Rod 10mm", "item_group": "Raw Materials", "stock_uom": "Kg", "is_stock_item": 1, "reorder_level": 50, "reorder_qty": 200},
    {"item_code": "RM-COPPER-002", "item_name": "Copper Wire 2mm", "item_group": "Raw Materials", "stock_uom": "Kg", "is_stock_item": 1, "reorder_level": 30, "reorder_qty": 100},
    {"item_code": "CP-VALVE-003", "item_name": "Hydraulic Valve Type B", "item_group": "Components", "stock_uom": "Nos", "is_stock_item": 1, "reorder_level": 10, "reorder_qty": 50},
    {"item_code": "CP-BEARING-004", "item_name": "Ball Bearing 6205", "item_group": "Components", "stock_uom": "Nos", "is_stock_item": 1, "reorder_level": 20, "reorder_qty": 100},
    {"item_code": "RM-ALUM-005", "item_name": "Aluminium Sheet 3mm", "item_group": "Raw Materials", "stock_uom": "Kg", "is_stock_item": 1, "reorder_level": 40, "reorder_qty": 150},
]

for item in items:
    create("Item", item)

# 3. Suppliers
suppliers = [
    {"supplier_name": "Acme Industrial Supplies", "supplier_group": "All Supplier Groups", "supplier_type": "Company"},
    {"supplier_name": "FastParts Co", "supplier_group": "All Supplier Groups", "supplier_type": "Company"},
    {"supplier_name": "MetalWorks Ltd", "supplier_group": "All Supplier Groups", "supplier_type": "Company"},
]

for supplier in suppliers:
    create("Supplier", supplier)

# 4. Warehouses
warehouses = [
    {"warehouse_name": "Main Store", "company": "ERPilot"},
    {"warehouse_name": "Raw Materials Store", "company": "ERPilot"},
]

for warehouse in warehouses:
    create("Warehouse", warehouse)

print("\n✅ Seed data complete!")