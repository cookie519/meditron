import pandas as pd

# Load the CSV file
file_name = 'train.csv'

RAREDISEASE = pd.read_csv(file_name, usecols=[1, 2, 3])
DISEASE = pd.read_csv(file_name, usecols=[1, 2, 4])
SYMPTOM = pd.read_csv(file_name, usecols=[1, 2, 5])
SIGN = pd.read_csv(file_name, usecols=[1, 2, 6])
print(RAREDISEASE.shape)
print(DISEASE.shape)
print(SYMPTOM.shape)
print(SIGN.shape)

# Drop rows where 'sign' column has NaN values
RAREDISEASE = RAREDISEASE.dropna(subset=['RAREDISEASE'])
DISEASE = DISEASE.dropna(subset=['DISEASE'])
SYMPTOM = SYMPTOM.dropna(subset=['SYMPTOM'])
SIGN = SIGN.dropna(subset=['SIGN'])
print(RAREDISEASE.shape)
print(DISEASE.shape)
print(SYMPTOM.shape)
print(SIGN.shape)

# rename column
RAREDISEASE = RAREDISEASE.rename(columns={'RAREDISEASE': 'response'})
DISEASE = DISEASE.rename(columns={'DISEASE': 'response'})
SYMPTOM = SYMPTOM.rename(columns={'SYMPTOM': 'response'})
SIGN = SIGN.rename(columns={'SIGN': 'response'})

# add column 'category'
RAREDISEASE['category'] = 'RAREDISEASE'
DISEASE['category'] = 'DISEASE'
SYMPTOM['category'] = 'SYMPTOM'
SIGN['category'] = 'SIGN'

# Concatenate
df = pd.concat([RAREDISEASE, DISEASE, SYMPTOM, SIGN], ignore_index=True)
df = df.rename(columns={'file_content': 'context'})

# Save the new DataFrame to a CSV file
df.to_csv(file_name[:-4]+'_out.csv', index=False)
