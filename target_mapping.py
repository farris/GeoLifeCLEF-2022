import pandas as pd

df = pd.read_csv('/scratch/fda239/Kaggle/data/observations/observations_fr_train_filter.csv')
n_unique = len(set(df.species_id))

species_unique = sorted(df['species_id'].unique())

species_map = {s : i for s,i in zip(species_unique,range(1,n_unique+1)) }

df['target_map'] = df.species_id.map(species_map)
print(df[['species_id','target_map']].sort_values(by='species_id'))