import numpy as np
import pandas as pd
#
# d = {'a':1, 'b': 2, 'c':3}
# print(pd.Series(d))
print(pd.Series(5.0, index=["a", "b", "c", "d", "e"]))

the_column = {'name': ['Juozas', 'Antanas', 'Petras' ],
              'Height': [15, 20, 22],
              'weight': [133, 124, 122]}
data = pd.DataFrame(the_column)
print(data)

#Value from Data Frame
svoris = data['weight'][2]
asmuo = data.iloc[2]
print(asmuo)

#Manipulate
bmi = []
for i in range(len(data)):
    bmi_score = data['weight'][i]/(data['Height'][i]**2)
    bmi.append(bmi_score)
data['bmi'] = bmi_score
print(data)

#To file
data.to_csv('my.csv', sep = '\t')