import pandas as pd
import numpy as np

url = "https://web.archive.org/web/20230902185326/https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"

tables = pd.read_html(url)

df = tables[3]

df = df.iloc[1:11, [0, 2]]
df.columns = ['Country', 'GDP (Million USD)']

df['GDP (Million USD)'] = (
    df['GDP (Million USD)']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace(r'\[.*\]', '', regex=True)
    .astype(float)
)

df['GDP (Billion USD)'] = np.round(df['GDP (Million USD)'] / 1000, 2)

df = df[['Country', 'GDP (Billion USD)']]

df.to_csv('Largest_economies.csv', index=False)

print(df)
