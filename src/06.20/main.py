import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('imdb_movies.csv')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
df = df.where(pd.notnull(df), 0)


