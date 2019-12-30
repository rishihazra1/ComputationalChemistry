import pandas as pd
import pickle as pk

data_path_prefix = "../data/"
molecule_prop = {}

df_molecule_prop = pd.read_csv(data_path_prefix + "molecular_energy.csv")
s = set(df_molecule_prop['molecule_name'].unique())
for molecule in s:
    molecule_prop[molecule] = df_molecule_prop.loc[df_molecule_prop['molecule_name']==molecule]
with open(data_path_prefix + "molecule_prop.pkl",'wb') as f:
    pk.dump(molecule_prop,f,pk.HIGHEST_PROTOCOL)
