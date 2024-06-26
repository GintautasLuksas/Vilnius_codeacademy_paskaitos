from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv
import re

def time_to_minutes(time_str):
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes

def extract_number(text):
    # Use regular expression to extract only digits
    number = re.search(r'\d+', text)
    if number:
        return int(number.group())  # Convert the matched number to an integer
    else:
        return 0  # Return 0 if no digits found

driver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=driver_service)

url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
driver.get(url)

# Find all movie elements
movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/a/h3')
movie_years = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[1]')
movie_rates_and_amounts = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span')
movie_lengths = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[2]')

# Ensure lengths match
min_length = min(len(movie_titles), len(movie_years), len(movie_rates_and_amounts), len(movie_lengths))

data = []
for i in range(min_length):
    title = movie_titles[i].text
    year = movie_years[i].text

    # Extract rate (rating value)
    rate_text = movie_rates_and_amounts[i].text.strip().split()[0]
    rate = float(rate_text) if rate_text else 0.0

    # Extract rate amount
    rate_amount_text = movie_rates_and_amounts[i].text.strip()
    rate_amount = extract_number(rate_amount_text)

    # Extract length if available
    if i < len(movie_lengths):
        length = time_to_minutes(movie_lengths[i].text)
    else:
        length = 0  # Set default value if length element is not found

    data.append([title, year, rate, rate_amount, length])

driver.quit()

csv_filename = 'imdb_movies.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Title', 'Year', 'Rating', 'Rating Amount', 'Duration (minutes)'])
    csvwriter.writerows(data)

print(f"IMDb movie data saved to {csv_filename}")
