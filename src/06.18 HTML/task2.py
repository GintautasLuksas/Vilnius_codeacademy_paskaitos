from lxml import html
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://www.meteo.lt/prognozes/lietuvos-miestai/?area=kaunas'
page = requests.get(url)

tree = html.fromstring(page.content)

days = tree.xpath('//div[@class="date"]/text()')


xpath_expr = '//table[@class="hourly-forecast" and @data-date="2024-06-19"]/tbody/tr/td/text()'

# Extract data using XPath
hourly = tree.xpath(xpath_expr)

# Print or process the extracted data
print(hourly)


