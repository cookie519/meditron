import pandas as pd
from difflib import SequenceMatcher

# Function to calculate similarity between two phrases
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

file_name = 'test_out1.csv'

# Load the CSV file
df = pd.read_csv(file_name)
for index, row in df.iterrows():
    # Access data in 'true' and 'generated' columns
    true_phrases = row['response'].split(', ')
    generated_phrases = row['output'].split(', ')
    df.at[index, 'gold_num'] = len(true_phrases)
    df.at[index, 'generated_num'] = len(generated_phrases)

    detected_count = 0
    # Compare each phrase in 'true' with all phrases in 'generated'
    for true_phrase in true_phrases:        
        for generated_phrase in generated_phrases:
            # Check if the similarity is 60% or more
            if similar(true_phrase, generated_phrase) >= 0.6:
                detected_count += 1
                break  # Stop checking once a match is found for this true_phrase
    df.at[index, 'TP'] = detected_count
    df.at[index, 'FN'] = len(true_phrases) - detected_count

for index, row in df.iterrows():
    # Access data in 'true' and 'generated' columns
    true_phrases = row['response'].split(', ')
    generated_phrases = row['output'].split(', ')

    detected_count = 0
    # Compare each phrase in 'true' with all phrases in 'generated'
    for generated_phrase in generated_phrases:       
        for true_phrase in true_phrases:
            if similar(true_phrase, generated_phrase) >= 0.6:
                detected_count += 1
                break  # Stop checking once a match is found for this true_phrase
    df.at[index, 'FP'] = len(generated_phrases) - detected_count


nums = {
    'RAREDISEASE': [0, 0, 0, 0, 0],
    'DISEASE': [0, 0, 0, 0, 0],
    'SIGN': [0, 0, 0, 0, 0],
    'SYMPTOM': [0, 0, 0, 0, 0]
}

# Create DataFrame
df_nums = pd.DataFrame(nums, index=['gold_num', 'generated_num', 'TP', 'FP', 'FN'])

scores = {
    'RAREDISEASE': [0, 0, 0],
    'DISEASE': [0, 0, 0],
    'SIGN': [0, 0, 0],
    'SYMPTOM': [0, 0, 0]
}

# Create DataFrame
df_scores = pd.DataFrame(scores, index=['precision', 'recall', 'f1'])

for category in ['RAREDISEASE', 'DISEASE', 'SIGN', 'SYMPTOM']:
    data =  df[df['category'] == category]
    gold_num = data['gold_num'].sum()
    generated_num = data['generated_num'].sum()
    TP = data['TP'].sum()
    FP = data['FP'].sum()
    FN = data['FN'].sum()

    df_nums[category] = [gold_num, generated_num, TP, FP, FN]

    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*pre*recall/(pre+recall)

    df_scores[category] = [pre, recall, f1]

df_nums['total'] = [df_nums.loc['gold_num'].sum(), df_nums.loc['generated_num'].sum(), df_nums.loc['TP'].sum(), df_nums.loc['FP'].sum(), df_nums.loc['FN'].sum()]


TP = df_nums['total']['TP'].sum()
FP = df_nums['total']['FP'].sum()
FN = df_nums['total']['FN'].sum()
pre = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = 2*pre*recall/(pre+recall)
df_scores['OVERALL'] = [pre, recall, f1]

df.to_csv(file_name, index=False)
df_nums.to_csv(file_name[:-4]+'nums.csv')
df_scores.to_csv(file_name[:-4]+'scores.csv')

print(df_nums)
print(df_scores)
