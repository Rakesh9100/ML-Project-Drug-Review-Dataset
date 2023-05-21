import pandas as pd

df = pd.read_table('datasets/drugsComTest_raw.tsv')

df.to_csv('datasets/testraw.csv')

df = pd.read_table('datasets/drugsComTrain_raw.tsv')

df.to_csv('datasets/trainraw.csv')

