from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv

# Function to convert time format (e.g., '1h 55m') to minutes
def time_to_minutes(time_str):
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes

# Set up Chrome WebDriver using WebDriver Manager (automatically installs the latest driver)
driver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=driver_service)

# IMDb URL to scrape
url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'

# Open IMDb URL in the browser
driver.get(url)

# Find elements using XPath to get movie titles, year, rating, and length
movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/a/h3')
movie_years = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[1]')
movie_rates = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span')
movie_lengths = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[2]')

# Determine the minimum length of the scraped lists
min_length = min(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths))

# Prepare data to write to CSV file
data = []
for i in range(min_length):
    title = movie_titles[i].text
    year = movie_years[i].text
    rate = movie_rates[i].text.split('\n')[0]  # Extract only the first number from rating
    length = time_to_minutes(movie_lengths[i].text)  # Convert time to minutes
    data.append([title, year, rate, length])

# Close the browser
driver.quit()

# Write data to CSV file
with open('imdb_movies.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Title', 'Year', 'Rating', 'Duration (minutes)'])  # Write header
    csvwriter.writerows(data)  # Write data rows

print("IMDb movie data saved to imdb_movies.csv")
