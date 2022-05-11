import pandas as pd

df = pd.read_csv("/scratch/fda239/Kaggle/data/observations/observations_fr_train_filter.csv")

print(df.columns)
print(len(df['species_id'].unique()))