import pubchempy as pcp
import pandas as pd
import random
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------
# STEP 1: Define drug list
# ------------------------------------------------

drug_list = [
"aspirin","ibuprofen","paracetamol","amoxicillin","ciprofloxacin",
"metformin","atorvastatin","simvastatin","amlodipine","losartan",
"omeprazole","pantoprazole","warfarin","clopidogrel"
]


# ------------------------------------------------
# STEP 2: Fetch chemical data from PubChem
# ------------------------------------------------

drug_data = []

print("Fetching drug structures...\n")

for drug in drug_list:

    try:

        compound = pcp.get_compounds(drug,'name')[0]
        smiles = compound.canonical_smiles

        mol = Chem.MolFromSmiles(smiles)

        MW = Descriptors.MolWt(mol)
        ExactMW = Descriptors.ExactMolWt(mol)

        LogP = Crippen.MolLogP(mol)
        MolMR = Crippen.MolMR(mol)

        TPSA = rdMolDescriptors.CalcTPSA(mol)

        HBD = Lipinski.NumHDonors(mol)
        HBA = Lipinski.NumHAcceptors(mol)

        RotBonds = Lipinski.NumRotatableBonds(mol)

        AromRings = rdMolDescriptors.CalcNumAromaticRings(mol)
        Rings = rdMolDescriptors.CalcNumRings(mol)

        HeavyAtoms = mol.GetNumHeavyAtoms()

        FractionCSP3 = rdMolDescriptors.CalcFractionCSP3(mol)

        ValenceElectrons = Descriptors.NumValenceElectrons(mol)

        HeteroAtoms = Descriptors.NumHeteroatoms(mol)

        MaxCharge = Descriptors.MaxPartialCharge(mol)
        MinCharge = Descriptors.MinPartialCharge(mol)

        DiffusionCoeff = 1/(MW**0.5)

        drug_data.append({

            "Drug":drug,
            "MW":MW,
            "ExactMW":ExactMW,
            "LogP":LogP,
            "MolMR":MolMR,
            "TPSA":TPSA,
            "HBD":HBD,
            "HBA":HBA,
            "RotBonds":RotBonds,
            "AromRings":AromRings,
            "Rings":Rings,
            "HeavyAtoms":HeavyAtoms,
            "FractionCSP3":FractionCSP3,
            "ValenceElectrons":ValenceElectrons,
            "HeteroAtoms":HeteroAtoms,
            "MaxCharge":MaxCharge,
            "MinCharge":MinCharge,
            "DiffusionCoeff":DiffusionCoeff
        })

        print("Fetched:",drug)

    except:

        print("Error fetching:",drug)


drug_df = pd.DataFrame(drug_data)

print("\nDrug Descriptor Dataset")
print(drug_df.head())


# ------------------------------------------------
# STEP 3: Generate synthetic patient dataset
# ------------------------------------------------

patients = []

sex = ["M","F"]
bp = ["LOW","NORMAL","HIGH"]
chol = ["NORMAL","HIGH"]

for i in range(500):

    drug = random.choice(drug_list)

    patients.append({

        "Age":random.randint(20,80),
        "Sex":random.choice(sex),
        "BP":random.choice(bp),
        "Cholesterol":random.choice(chol),
        "Na_to_K":random.randint(10,35),
        "Drug":drug
    })


patient_df = pd.DataFrame(patients)

print("\nPatient Dataset Sample")
print(patient_df.head())


# ------------------------------------------------
# STEP 4: Merge patient + chemical features
# ------------------------------------------------

df = pd.merge(patient_df,drug_df,on="Drug")

print("\nMerged Dataset")
print(df.head())


# ------------------------------------------------
# STEP 5: Encode categorical features
# ------------------------------------------------

le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["BP"] = le_bp.fit_transform(df["BP"])
df["Cholesterol"] = le_chol.fit_transform(df["Cholesterol"])
df["Drug"] = le_drug.fit_transform(df["Drug"])


# ------------------------------------------------
# STEP 6: Train Test Split
# ------------------------------------------------

X = df.drop("Drug",axis=1)
y = df["Drug"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)


# ------------------------------------------------
# STEP 7: Train Machine Learning Model
# ------------------------------------------------

model = RandomForestClassifier(
n_estimators=500,
max_depth=12,
random_state=42
)

model.fit(X_train,y_train)

print("\nModel Training Completed")


# ------------------------------------------------
# STEP 8: Evaluate Model
# ------------------------------------------------

y_pred = model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,y_pred))


# ------------------------------------------------
# STEP 9: Feature Importance
# ------------------------------------------------

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,7))
sns.barplot(x=importance,y=features)
plt.title("Feature Importance (Chemical + Patient Features)")
plt.show()


# ------------------------------------------------
# STEP 10: Predict Drug for New Patient
# ------------------------------------------------

new_patient = pd.DataFrame({

"Age":[55],
"Sex":["F"],
"BP":["HIGH"],
"Cholesterol":["HIGH"],
"Na_to_K":[28],
"MW":[180],
"ExactMW":[180],
"LogP":[1.2],
"MolMR":[45],
"TPSA":[63],
"HBD":[1],
"HBA":[3],
"RotBonds":[2],
"AromRings":[1],
"Rings":[1],
"HeavyAtoms":[13],
"FractionCSP3":[0.2],
"ValenceElectrons":[70],
"HeteroAtoms":[4],
"MaxCharge":[0.3],
"MinCharge":[-0.4],
"DiffusionCoeff":[1/(180**0.5)]
})


new_patient["Sex"] = le_sex.transform(new_patient["Sex"])
new_patient["BP"] = le_bp.transform(new_patient["BP"])
new_patient["Cholesterol"] = le_chol.transform(new_patient["Cholesterol"])


prediction = model.predict(new_patient)

drug_name = le_drug.inverse_transform(prediction)

print("\nRecommended Drug:",drug_name[0])