# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
############################ 2023 #################################################
usagers_2023 = pd.read_csv("../data/usagers-2023.csv", sep=";").set_index("Num_Acc")
lieux_2023 = pd.read_csv("../data/lieux-2023.csv", sep=";").set_index("Num_Acc")
vehicules_2023 = pd.read_csv("../data/vehicules-2023.csv", sep=";").set_index("Num_Acc")
caracteristiques_2023 = pd.read_csv("../data/caract-2023.csv", sep=";").set_index("Num_Acc")
############################ 2022 #################################################
usagers_2022 = pd.read_csv("../data/usagers-2022.csv", sep=";").set_index("Num_Acc")
lieux_2022 = pd.read_csv("../data/lieux-2022.csv", sep=";").set_index("Num_Acc")
vehicules_2022 = pd.read_csv("../data/vehicules-2022.csv", sep=";").set_index("Num_Acc")
caracteristiques_2022 = pd.read_csv("../data/carcteristiques-2022.csv", sep=";").rename(columns={"Accident_Id": "Num_Acc"}).set_index("Num_Acc")
############################ 2021 #################################################
usagers_2021 = pd.read_csv("../data/usagers-2021.csv", sep=";").set_index("Num_Acc")
lieux_2021 = pd.read_csv("../data/lieux-2021.csv", sep=";").set_index("Num_Acc")
vehicules_2021 = pd.read_csv("../data/vehicules-2021.csv", sep=";").set_index("Num_Acc")
caracteristiques_2021 = pd.read_csv("../data/carcteristiques-2021.csv", sep=";").set_index("Num_Acc")


# %%
usagers = pd.concat([usagers_2021, usagers_2022, usagers_2023], axis=0)
lieux = pd.concat([lieux_2021, lieux_2022, lieux_2023], axis=0)
vehicules = pd.concat([vehicules_2021, vehicules_2022, vehicules_2023], axis=0)
caracteristiques = pd.concat([caracteristiques_2021, caracteristiques_2022, caracteristiques_2023], axis=0)

# %%
print(vehicules.columns)
print(usagers.columns)
print(lieux.columns)
print(caracteristiques.columns)

# %%
index = usagers.index.unique()

print(all(usagers.index.unique() == index))
print(all(lieux.index.unique() == index))
print(all(vehicules.index.unique() == index))
print(all(caracteristiques.index.unique() == index))

# %%
df = usagers.merge(vehicules, on=["Num_Acc", "id_vehicule"], how="outer") \
            .merge(lieux, on="Num_Acc", how="outer") \
            .merge(caracteristiques, on="Num_Acc", how="outer")


# %%
print(df.columns)

# %%
# Compute correlation matrix for numerical columns only
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(16, 10))
sns.heatmap(corr,cmap="coolwarm")#annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %%
# Compute correlation matrix for numerical columns only
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(16, 10))
sns.heatmap(corr,cmap="coolwarm")#annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %%
# df.to_csv("../data/accidents_2021-2023.csv", index=False)

# %%
