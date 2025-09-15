import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text

DB = os.getenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpass@db:5432/ragdb")
OUT_DIR = os.getenv("OUT_DIR", "/data-gen/out")

engine = create_engine(DB, pool_pre_ping=True)

DDL = """
CREATE SCHEMA IF NOT EXISTS ins;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
"""

# Declare expected date columns per table (edit if your generator changes).
DATE_COLS = {
    "customers": ["dob"],
    "policies": ["start_date", "end_date"],
    "coverages": [],
    "properties": [],
    "rental_units": [],
    "vehicles": [],
    "claims": ["loss_date", "report_date", "close_date"],
    "loss_events": ["loss_date", "report_date", "close_date"],
    "geo_features": [],
}

TABLE_ORDER = [
    "customers","policies","coverages","properties","rental_units",
    "vehicles","claims","loss_events","geo_features"
]

def read_csv_safely(path: str, table: str) -> pd.DataFrame:
    # Only parse date columns that actually exist in this CSV
    present = [c for c in DATE_COLS.get(table, [])]
    # First read without parse_dates to avoid pandas complaining about missing columns
    df = pd.read_csv(path, keep_default_na=True, na_values=[""])
    # Now convert present date columns if they exist
    for col in present:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df

def upsert(df: pd.DataFrame, name: str):
    df.to_sql(name, engine, schema="ins", if_exists="replace", index=False)

def add_indexes(conn):
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_claims_loss_date ON ins.claims (loss_date);",
        "CREATE INDEX IF NOT EXISTS idx_claims_product_peril ON ins.claims (product_type, peril);",
        "CREATE INDEX IF NOT EXISTS idx_policies_customer ON ins.policies (customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_loss_events_geo ON ins.loss_events (county_code, city);"
    ]
    for s in stmts:
        conn.execute(text(s))

if __name__ == "__main__":
    with engine.begin() as conn:
        conn.execute(text(DDL))

    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(OUT_DIR, "*.csv"))}

    for t in TABLE_ORDER:
        p = files.get(t)
        if not p:
            print(f"SKIP: {t}.csv not found")
            continue
        print(f"Loading {t} …")
        df = read_csv_safely(p, t)
        upsert(df, t)

    with engine.begin() as conn:
        add_indexes(conn)

    print("Load done.")
