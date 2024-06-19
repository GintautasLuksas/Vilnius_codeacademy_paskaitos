from lxml import html
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://www.meteo.lt/?area=kaunas'
page = requests.get(url)

tree = html.fromstring(page.content)
days = tree.xpath('//div[@class="day-wrap"]/h4/text()')

day_list = [i.strip() for i in days]


temps = tree.xpath('//div[@class="temprature"]/text()')
temp_list = []
for i in temps[:7]:
    value = re.match(r'^\d+', i.strip())
    if value:
        temp_list.append(int(value.group()))


night_temp_list = [temp_list[0]]
night_temp = tree.xpath('//div[@class="night-weather"]/span/text()')
for i in night_temp:
    value = re.match(r'^\d+', i.strip())
    if value:
        night_temp_list.append(int(value.group()))


if len(day_list) == len(temp_list) == len(night_temp_list):
    df = pd.DataFrame({
        'Sav_diena': day_list,
        'Dienos_temp': temp_list,
        'Nakties_temp': night_temp_list
    })

df.to_csv('output.csv', index=False)

plt.figure(figsize=(8, 4))
plt.plot(df['Sav_diena'], df['Dienos_temp'], marker='o', linestyle='-', color='y')
plt.title('Dienos temperatura per savaite')
plt.xlabel('Savaitės diena')
plt.ylabel('Dienos temperatūra (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()


plt.figure(figsize=(8, 4))
plt.plot(df['Sav_diena'], df['Nakties_temp'], marker='o', linestyle='-', color='y')
plt.title('Nakties temperatura per savaite')
plt.xlabel('Savaitės diena')
plt.ylabel('Nakties Temperatūra (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()