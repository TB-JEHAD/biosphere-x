import pandas as pd
import re

def normalize(col):
    return re.sub(r"[^\w]", "", col.lower())

df = pd.read_excel('mars_data.xlsx')
print('Temperature columns and their normalized names:')
for col in df.columns:
    if 'temp' in col.lower():
        print(f'  "{col}" -> "{normalize(col)}"')

print('\nLooking for temperature metric:')
normalized_metric = normalize("temperature")
print(f'Looking for: "{normalized_metric}"')

metric_mapping = {
    "temperature": ["mintemperaturec", "maxtemperaturec"],
}

if normalized_metric in metric_mapping:
    print(f'Found mapping: {metric_mapping[normalized_metric]}')
    for target in metric_mapping[normalized_metric]:
        for col in df.columns:
            if normalize(col) == target:
                print(f'  MATCH: "{col}" -> "{target}"')
                break
