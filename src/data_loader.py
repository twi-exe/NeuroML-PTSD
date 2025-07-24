import pandas as pd

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    ptsd = df[df['specific.disorder'] == 'Posttraumatic stress disorder']
    hc = df[df['specific.disorder'] == 'Healthy control'].sample(n=52, random_state=42)
    data = pd.concat([ptsd, hc])
    data.drop(columns=['no.', 'eeg.date', 'main.disorder', 'Unnamed: 122', 'age', 'education', 'sex'], inplace=True)
    data['specific.disorder'] = data['specific.disorder'].map({'Healthy control': 0, 'Posttraumatic stress disorder': 1})
    return data
