
import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

output_file = Path("electricity_consumption_100MB.csv")
target_size = 100_000_000  # Approximately 100 MB

random.seed(42)
start = datetime(2024, 1, 1)
rows = 0

with output_file.open(
    "w", newline="", encoding="utf-8", buffering=1024 * 1024
) as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["timestamp", "demand"])

    while file.tell() < target_size:
        timestamp = start + timedelta(seconds=rows)

        daily_pattern = 120 * (
            1 + math.sin((timestamp.hour - 7) * math.pi / 12)
        )
        weekend_factor = -70 if timestamp.weekday() >= 5 else 0
        noise = random.uniform(-45, 45)

        demand = max(
            100,
            850 + daily_pattern + weekend_factor + noise
        )

        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{demand:.2f}"
        ])
        rows += 1

print("Dataset created:", output_file.resolve())
print("Rows generated:", rows)
print("Columns: timestamp, demand")
print("Target size: approximately 100 MB")
