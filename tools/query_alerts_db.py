import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Query alerts.db for a given date")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parents[1] / "src" / "alerts.db"),
        help="Path to alerts.db (default: <repo>/src/alerts.db)",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date prefix in YYYY-MM-DD (matches ISO timestamps stored in the DB)",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2

    like_pattern = f"{args.date}%"

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (like_pattern,))
    count = cur.fetchone()[0]
    print("db:", db_path)
    print("date:", args.date)
    print("rows:", count)

    cur.execute(
        "SELECT id, timestamp, src_ip, action, details FROM alerts "
        "WHERE timestamp LIKE ? ORDER BY id DESC LIMIT ?",
        (like_pattern, int(args.limit)),
    )
    rows = cur.fetchall()
    print("recent:")
    for row in rows:
        print(" ", row)

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
