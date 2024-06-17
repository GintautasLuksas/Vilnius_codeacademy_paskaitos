import pandas as pd

#1
df = pd.read_csv(r'C:\Users\BossJore\PycharmProjects\Vilnius_codeacademy_paskaitos\src\06.17 pandas\df_pandas_example.csv')
# print(df)
#2 2. Parodykite pirmąsias 5 DataFrame eilutes, 10 DataFrame eilučių.
# print(df.head(10))
# 3. Parodykite paskutines 5 DataFrame eilutes, 10 DataFrame eilučių.

# 4. Atspausdinkite informaciją apie DataFrame.
# print(df.info())

# 5. Išrinkite stulpelius "Name" (vardas) ir "City" (miestas).
# selected = df[['Name', 'City']]
# print(selected)

# 6. Išrinkite eilutes, kuriose " Age" yra didesnis nei 30 metų.
# filtered = df[df['Age'] > 30]
# print(filtered)

# 7. Išrinkite eilutes, kuriose "'City" yra "San Francisco" ir "Salary" yra didesnis nei 80000
# filtered2 = df[(df['City'] == 'San Francisco') & (df['Salary'] > 80000)]
# print(filtered2)

# # 8. Pridėkite naują stulpelį " Bonus", kuris yra 10 % " Salary" ("Atlyginimas")
# df['Bonus'] = df['Salary'] * 0.1
# print(df)
#
# # 9. Pervadinkite stulpelį " Bonus" į "EmployeeBonus".
# df = df.rename(columns= {'Bonus': 'EmployeeBonus'})
# print(df)
#
# # 10. Pašalinkite stulpelį "EmployeeBonus".
# df.drop(columns=['EmployeeBonus'], inplace=True)
# print(df)

# 11. Rūšiuokite duomenų rėmelį pagal "Age" mažėjančia tvarka
# df_sorted = df.sort_values(by='Age', ascending=False)
# print(df_sorted)
# 12. Sugrupuokite DataFrame pagal "City" ir apskaičiuokite "Salary" vidurkį
# df_sorted2 = df.sort_values(by='City')
# df_average = df_sorted2['Salary'].mean()
# print(df_sorted2)
# print(f'Average salary is {df_average}.')

# # 13. Sugrupuokite DataFrame pagal " Department" ir gaukite kiekvienos grupės dydį
# df_sorted3 = df.groupby('Department').size()
# print(df_sorted3)


# # 14. Raskite didžiausią "Salary" kiekviename "City" (mieste)
# df_max = df.groupby('City')['Salary'].max()
# print(df_max)

# # 15. Raskite mažiausią "Age" kiekviename "Department".
# df_amin = df.groupby('Department')['Age'].min()
# print(df_amin)

# # 16. Suskaičiuokite darbuotojų skaičių kiekviename "City".
# df_counted = df.groupby('City').size()
# print(df_counted)

# # 17. Užpildykite trūkstamas reikšmes duomenų rėmelyje (jei tokių yra) reikšme "Unknown".
# missing_info = df.isnull().sum()
# df_filled = df.fillna('Unknown')
# print(missing_info)


# # 18. Išsaugokite DataFrame į naują CSV failą, pavadintą "employees_updated.csv".
# df.to_csv('employees_updated.csv', index = False)
# 19. Filtruokite DataFrame, kad būtų rodomi tik "Engineering" skyriaus darbuotojai

# filtered_data = df[df['Department'] == 'Engineering']
# print(filtered_data)
#
#
# 20. Normalizuokite stulpelį "Salary" naudodami Min-Max skalę ir pridėkite jį kaip naują stulpelį Normalized_Salary:
# df['Normalized Salary'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
# # print(df)
# # 21. Sukurkite naują DataFrame, kuriame būtų rodomi daugiausiai uždirbantys kiekvieno skyriaus darbuotojai
# df_rich = df.sort_values(by='Salary').groupby('Department').head(1)
# print(df_rich)
# 22. Nustatykite koreliaciją tarp "Age" ir "Salary".
print(df['Age'].corr(df['Salary']))
# df['Rate'] = df['Salary'] / df['Age']
# df_sorted2 = df.sort_values(by='Age'), df[['Age', 'Rate']]
# print(df_sorted2)


# # 23. Sukurkite naują stulpelį Age_Group, skirstantį darbuotojus į "Jaunus" (<30), "Vidutinio amžiaus" (30-40) ir "Vyresnius" (>40):
# def age_category(age):
#     if age < 30:
#         return 'Jaunas'
#     elif age == 30 or age < 40:
#         return 'Vidutinio amziaus'
#     elif age > 40:
#         return 'Vyersnis'
#
#
# df['Age_Group'] = df['Age'].apply(age_category)
# # print(df)
#
# # 24. Kiekvienam skyriui apskaičiuokite skirtumą tarp kiekvieno darbuotojo darbo užmokesčio ir skyriaus vidutinio darbo užmokesčio
# df_engineering = df[df['Department'] == 'Engineering']
# df_engineering_avg = df_engineering['Salary'].mean()
# df_engineering['Avg_diff'] = df_engineering['Salary'] - df_engineering_avg
# print(df_engineering)
#
#
# df_HR = df[df['Department'] == 'HR']
# df_HR_avg = df_HR['Salary'].mean()
# df_HR['Avg_diff'] = df_HR['Salary'] - df_HR_avg
# print(df_HR)
#
#
# df_Marketing = df[df['Department'] == 'Marketing']
# df_Marketing_avg = df_Marketing['Salary'].mean()
# df_Marketing['Avg_diff'] = df_Marketing['Salary'] - df_Marketing_avg
# print(df_Marketing)
#
# # Calculate average salary for each department
# department_avg_salary = df.groupby('Department')['Salary'].transform('mean')
#
# # Calculate salary difference for each employee within their department
# df['Avg_diff'] = df['Salary'] - department_avg_salary
#
# # Display the updated DataFrame with salary differences
# print(df)