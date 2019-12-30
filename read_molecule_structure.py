import pandas as pd
import pickle as pk
from mendeleev import get_table
ptable = get_table('elements')
import os

set_of_molecules = set()

ptable_access = {}
for _,row in ptable.iterrows():
    ptable_access[row["symbol"]] = row["atomic_number"]

data_path_prefix = "../data/"
path_to_structures = data_path_prefix + "/structures/"

molecule_structures = {}

for i,molecule in enumerate(os.listdir(path_to_structures)):
    df = pd.read_csv(path_to_structures + molecule, sep = ' ', skiprows= [0],names = ["atom","x","y","z"])
    for a_index in range(len(df)):
        atom = df["atom"].iloc[a_index]
        df.at[a_index, "num_protons"] = ptable_access[atom]
    molecule_structures[molecule[:-4]] = df
    set_of_molecules.add(molecule[:-4])

print("Pickling")
with open(data_path_prefix + "molecule_composition_data.pkl",'wb') as f:
    pk.dump(molecule_structures,f,pk.HIGHEST_PROTOCOL)
with open(data_path_prefix + "set_of_molecules.pkl",'wb') as f:
    pk.dump(set_of_molecules,f,pk.HIGHEST_PROTOCOL)
