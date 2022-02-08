import pickle
import pandas as pd
with open("test.pkl", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

# cols = []
# for key in b[0].keys():
#     cols.append(key)
# print(cols)

df = pd.DataFrame.from_dict(b[0], orient='index').transpose()
for idx, data in enumerate(b):
    if idx == 0:
        continue
    else:
        new = pd.DataFrame.from_dict(b[idx], orient='index').transpose()
        df = pd.concat([df, new])
df = df.reset_index(drop=True)
df['creation_timestamp'] = pd.to_datetime(df['creation_timestamp'], unit='ms')
df['expiration_timestamp'] = pd.to_datetime(df['expiration_timestamp'], unit='ms')
df.to_csv("instruments.csv", index=False)
print(df)

