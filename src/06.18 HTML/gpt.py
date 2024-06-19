from lxml import html
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

# Fetch the web page
url = 'https://www.meteo.lt'
page = requests.get(url)
print(page.content)

# Parse the HTML content
tree = html.fromstring(page.content)

# Extract days
days = tree.xpath('//div[@class="day-wrap"]/h4/text()')
print(days)
# day_list = [i.strip() for i in days]
#
# # Extract dates
# dates = tree.xpath('//div[@class="date"]/text()')
# print(dates)
# date_list = [i.strip() for i in dates]
#
# # Extract temperatures
# temps = tree.xpath('//div[@class="temprature"]/text()')
# print(temps)
# temp_list = []
# for i in temps:
#     value = re.match(r'^\d+', i.strip())
#     if value:
#         temp_list.append(int(value.group()))
# print(temp_list)
#
# # Extract wind data
# winds = tree.xpath('//div[@class="wind"]/text()')
# print(winds)
# wind_list = [i.strip() for i in winds]
#
# wind_speed = []
# wind_direction = []
# for i in wind_list:
#     splitted = i.split()
#     if len(splitted) >= 2:
#         wind_speed.append(splitted[0])
#         wind_direction.append(splitted[1])
# print(wind_speed)
# print(wind_direction)
#
# # Create DataFrame
# df = pd.DataFrame({
#     'Data': date_list,
#     'Sav_diena': day_list,
#     'Temperatura': temp_list,
#     'Vejas': wind_speed,
#     'Vejas_kryptis': wind_direction
# })
# print(df)
#
# #
#
# # Plot data
# plt.figure(figsize=(10, 6))
# plt.plot(df['Data'], df['Temperatura'], marker='o', linestyle='-', color='b')
# plt.title('Temperatura per savaite')
# plt.xlabel('Data')
# plt.ylabel('Temperatura (°C)')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # Function to calculate average wind speed
# def avg_wind_speed(wind_speed):
#     splitted = wind_speed.split('-')
#     if len(splitted) == 2:
#         return (int(splitted[0]) + int(splitted[1])) / 2
#     return int(wind_speed)
#
# # Apply average wind speed calculation
# df['Vejas_greitis_avg'] = df['Vejas'].apply(avg_wind_speed)
# print(df)
#
# # Plot average wind speed
# plt.figure(figsize=(8, 4))
# plt.bar(df['Data'], df['Vejas_greitis_avg'], color='r')
# plt.xlabel('Data')
# plt.ylabel('Vidutinis vejas (m/s)')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # Plot wind direction distribution
# wind_direction_counts = df['Vejas_kryptis'].value_counts()
# print(wind_direction_counts)
# plt.figure(figsize=(6, 6))
# plt.pie(wind_direction_counts, labels=wind_direction_counts.index, autopct='%1.1f%%')
# plt.title('Vejo kryptis')
# plt.show()
