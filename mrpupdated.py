import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor


# ------------------------------------------------
# STEP 1: FETCH DRUGS (API)
# ------------------------------------------------

print("Fetching drugs...")

all_data = []

for offset in range(0, 300, 100):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&limit=100&offset={offset}"

    try:
        session = requests.Session()
        response = session.get(url, timeout=10)
        data = response.json()
        all_data.extend(data['molecules'])
    except:
        continue


# ------------------------------------------------
# STEP 2: RDKit FEATURES
# ------------------------------------------------

drug_data = []

for mol in all_data:

    try:
        name = mol.get("pref_name")
        struct = mol.get("molecule_structures")

        if not name or not struct:
            continue

        smiles = struct.get("canonical_smiles")
        if not smiles:
            continue

        mol_obj = Chem.MolFromSmiles(smiles)

        drug_data.append({
            "Drug": name,
            "MW": Descriptors.MolWt(mol_obj),
            "LogP": Crippen.MolLogP(mol_obj),
            "TPSA": rdMolDescriptors.CalcTPSA(mol_obj)
        })

    except:
        continue


drug_df = pd.DataFrame(drug_data).drop_duplicates()
drug_df.to_csv("final_drugs.csv", index=False)

print("CSV saved successfully")
drug_df = drug_df[
    (drug_df["MW"] > 150) & 
    (drug_df["MW"] < 600) &
    (drug_df["LogP"] > -2) & 
    (drug_df["LogP"] < 6) &
    (drug_df["TPSA"] < 150)
]

drug_df = drug_df[
    ~drug_df["Drug"].str.contains("acid|sodium|chloride|glycine", case=False)
]

# ------------------------------------------------
# STEP 3: SAMPLE PATIENTS
# ------------------------------------------------

patients = [
    {"Age": 25, "BP": "NORMAL", "Cholesterol": "HIGH", "Na_to_K": 15, "Symptoms": ["fever","pain"]},
    {"Age": 50, "BP": "HIGH", "Cholesterol": "HIGH", "Na_to_K": 25, "Symptoms": ["chest pain"]},
    {"Age": 35, "BP": "LOW", "Cholesterol": "NORMAL", "Na_to_K": 20, "Symptoms": ["infection"]},
    {"Age": 60, "BP": "HIGH", "Cholesterol": "NORMAL", "Na_to_K": 30, "Symptoms": ["headache"]},
    {"Age": 45, "BP": "NORMAL", "Cholesterol": "HIGH", "Na_to_K": 22, "Symptoms": ["pain"]}
]


# ------------------------------------------------
# STEP 4: RULE-BASED SCORE (FOR TRAINING)
# ------------------------------------------------

def rule_score(patient, drug):

    score = 0

    score -= abs(drug["TPSA"] - patient["Na_to_K"]*3)
    score -= abs(drug["MW"] - patient["Age"]*3)

    if patient["BP"] == "HIGH":
        score -= abs(drug["LogP"] - 2.5)

    if patient["Cholesterol"] == "HIGH":
        score -= abs(drug["LogP"] - 3)

    return score


# ------------------------------------------------
# STEP 5: CREATE TRAINING DATA
# ------------------------------------------------

train_data = []

for patient in patients:
    for _, drug in drug_df.iterrows():

        train_data.append({
            "MW": drug["MW"],
            "LogP": drug["LogP"],
            "TPSA": drug["TPSA"],
            "Age": patient["Age"],
            "Na_to_K": patient["Na_to_K"],
            "Score": rule_score(patient, drug)
        })

train_df = pd.DataFrame(train_data)


# ------------------------------------------------
# STEP 6: TRAIN ML MODEL
# ------------------------------------------------

X = train_df.drop("Score", axis=1)
y = train_df["Score"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

print("ML model trained")


# ------------------------------------------------
# STEP 7: PREDICTION + RANKING
# ------------------------------------------------

# ------------------------------------------------
# STEP 7: PREDICTION + RANKING + REASONS
# ------------------------------------------------

for idx, patient in enumerate(patients):

    print(f"\n==============================")
    print(f"Patient {idx+1}: {patient}")
    print("==============================")

    # ----------------------------
    # VECTORIZED ML PREDICTION
    # ----------------------------

    features = pd.DataFrame({
        "MW": drug_df["MW"],
        "LogP": drug_df["LogP"],
        "TPSA": drug_df["TPSA"],
        "Age": patient["Age"],
        "Na_to_K": patient["Na_to_K"]
    })

    drug_df["Score"] = model.predict(features)

    # ----------------------------
    # SORT
    # ----------------------------

    ranked = drug_df.sort_values(by="Score", ascending=False)
    top5 = ranked.head(5)

    print("\nTop 5 Recommended Drugs:\n")

    # ----------------------------
    # EXPLANATION (ONLY FOR TOP 5)
    # ----------------------------

    for _, row in top5.iterrows():

        reasons = []

        if abs(row["TPSA"] - patient["Na_to_K"]*3) < 30:
            reasons.append("Good polarity match")

        if abs(row["MW"] - patient["Age"]*3) < 100:
            reasons.append("Suitable diffusion")

        if patient["BP"] == "HIGH" and abs(row["LogP"] - 2.5) < 1:
            reasons.append("BP compatibility")

        if patient["Cholesterol"] == "HIGH" and abs(row["LogP"] - 3) < 1:
            reasons.append("Lipid control")

        if "pain" in patient["Symptoms"]:
            reasons.append("Pain relief suitability")

        if "infection" in patient["Symptoms"]:
            reasons.append("Infection targeting")

        if not reasons:
            reasons.append("General compatibility")

        print(f"{row['Drug']} → Score: {round(row['Score'],2)}")
        print(f"Reason: {', '.join(reasons)}")
        print("-"*40)

    # Graph
    plt.figure()
    plt.barh(top5["Drug"], top5["Score"])
    plt.title(f"Top Drugs (Patient {idx+1})")
    plt.xlabel("Predicted Score")
    plt.ylabel("Drug")

    #plt.show()


# ------------------------------------------------
# STEP 8: GLOBAL GRAPH
# ------------------------------------------------

plt.figure()
plt.hist(drug_df["LogP"], bins=30)
plt.title("LogP Distribution")
plt.xlabel("LogP")
plt.ylabel("Frequency")
plt.show()