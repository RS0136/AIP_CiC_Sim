#!/usr/bin/env python3
import sys, pandas as pd
if len(sys.argv) < 2:
    print("Usage: python src/inspect_csv.py data/filteredCorpus.csv")
    sys.exit(1)
df = pd.read_csv(sys.argv[1], nrows=5)
print("Columns:", list(df.columns))
print(df.head(3).T)
