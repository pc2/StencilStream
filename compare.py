import pandas as pd
import numpy as np

epsilon = 1e-6  # tolerance threshold

base_file = "hz.172592.csv"
prot_file = "hz.172592base.csv"

try:
    df_base = pd.read_csv(base_file)
    df_prot = pd.read_csv(prot_file)

    # check if shapes are the same
    if df_base.shape != df_prot.shape:
        print(f"{base_file}: ❌ Files have different shapes: {df_base.shape} vs {df_prot.shape}")
    else:
        # compare with tolerance
        diffs = ~np.isclose(df_base.values, df_prot.values, atol=epsilon, equal_nan=True)

        if not diffs.any():
            print(f"{base_file}: ✅ Files are equal within tolerance (ε = {epsilon})")
        else:
            print(f"{base_file}: ❌ Files DIFFER within tolerance ε = {epsilon}")
            rows, cols = np.where(diffs)
            for row, col in zip(rows, cols):
                val_base = df_base.iat[row, col]
                val_prot = df_prot.iat[row, col]
                col_name = df_base.columns[col]
                print(f"  ↪ Row {row}, Column '{col_name}': base={val_base} ≠ prot={val_prot}")

except FileNotFoundError as e:
    print(f"❗ File missing - {e.filename}")
