from lxml import html
import requests

url = 'https://www.meteo.lt'
page = requests.get(url)
print(page.content)

tree = html.fromstring(page.content)
days = tree.xpath('//div[@class="day-wrap"]/h4/text()')
print(days)
day_list = []
for i in days:
    day_list.append(i.strip)))


date = tree.xpath('//div[@class="date"]/text()')
print(date)
date_list = []
for i in days:
    date_list.append(i.strip)))



temp = tree.xpath('//div[@class='temprature']/text()')
print(temp)
temp_list = []
for i in temp:
    value = re.match(r'^\d+', i.strip())
    temp_list.append(int(value.group())
print(temp_list)

wind =  '//div[@class='wind']/text()')
print(wind)
wind_list = []
for i in wind:
    wind_list.append(i.strip())

wind_speed = []
wind_direction = []
for i inv wind_list:
    splitted = i.split()
    if len(splitted) == 3:
        wind_speed.append(splitted[0])
        wind_direction.append(splitted[2])
print(wind_speed)
print(wind_direction)

import pandas as pd
df = pd.DataFrame({
    'Data': date_list,
    'Sav_diena': days_list,
    'Temperatura': temp_list,
    'Vejas': wind_speed
    'Vejas_kryptis': wind_direction
})
print(df)

df.to_csv('output.csv')
import matplotlib.pyplot as plt
pit.figure(figsize=(10, 6))
plt.plot(df['Data', df['Temperatura'], marker='o', linestyle='-', color='b')
plt.title('Temp per savaite')
plt.xlabel('Data')
plt.ylabel('Temp C')
plt.grid(True)
plt.show()

def avg_wind_speed(wind_speed):
    splitted = wind_speed.split('-')
    return(int.splitted(0)+int(splitted(1))/2)

df['Vejas_greitis_avg'] = df['Vejas'].apply()
print(df)

plt.figure(figsize=(8, 4))
plt.bar(df['Data'], df['Vejas'], color='r')
plt.xlabel('Data')
plt.ylabel('Vejas')
plt.grid(True)
plt.show()

wind_direction_counts = df['Vejas_kryptis'].value_counts()
print(wind_drection_counts)
plt.figure(figsize =(6,6))
plt.pie(wind_direction_counts, label=wind_drection_counts.index, autopct = '%1.1f%%')
plt.title('Vejo kryptis')
plt.show()