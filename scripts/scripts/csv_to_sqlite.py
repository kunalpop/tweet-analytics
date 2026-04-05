import csv
import sqlite3
import os
import sys


def csv_to_sqlite(csv_file="../data/tweets.csv", db_file="../data/tweetsDB.sqlite", table_name="tweets"):
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found.")
        sys.exit(1)

    # Read CSV file
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    if not columns:
        print("Error: CSV file is empty or has no headers.")
        sys.exit(1)

    print(f"Found {len(rows)} rows and {len(columns)} columns: {columns}")

    # Connect to SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop table if it already exists and recreate it
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create table — all columns as TEXT for flexibility
    col_defs = ", ".join(f'"{col}" TEXT' for col in columns)
    cursor.execute(f"CREATE TABLE {table_name} ({col_defs})")

    # Insert rows
    placeholders = ", ".join("?" for _ in columns)
    col_names = ", ".join(f'"{col}"' for col in columns)
    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

    for row in rows:
        values = [row.get(col) for col in columns]
        cursor.execute(insert_sql, values)

    conn.commit()
    conn.close()

    print(f"✅ Done! Database saved to '{db_file}', table '{table_name}'.")
    print(f"   Inserted {len(rows)} rows.")


if __name__ == "__main__":
    csv_to_sqlite()
