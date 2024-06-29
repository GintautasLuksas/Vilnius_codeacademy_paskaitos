from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up ChromeOptions with headless mode
chrome_options = Options()
chrome_options.headless = False  # Set to True if you want headless mode

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

try:
    # Navigate to the URL
    url = 'https://autoplius.lt/skelbimai/naudoti-automobiliai/audi?category_id=2&make_date_from=1990&make_date_to=2010&slist=2315347586'
    driver.get(url)

    # Wait for the car listings to load
    wait = WebDriverWait(driver, 20)  # Increase wait time if necessary
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.item')))

    # Find all car listing elements
    car_listings = driver.find_elements(By.CSS_SELECTOR, '.item')

    data = []
    for car in car_listings:
        try:
            # Explicitly wait for the price element to be visible within each car listing
            price_element = WebDriverWait(car, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, '.price-block span.price'))
            )
            price = price_element.text.strip()
            data.append(price)
        except Exception as e:
            print(f"Error extracting price for a car: {e}")

    # Print all prices found
    for price in data:
        print(f"Price: {price}")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Ensure the WebDriver is properly closed
    if driver:
        driver.quit()
