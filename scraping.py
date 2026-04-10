import random
import re
import time
import logging

from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlencode
from selenium import webdriver

logging.basicConfig(level=logging.DEBUG, filename="./logs/ta_scraping.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")

BASE_URL = 'https://www.tripadvisor.ru/FindRestaurants'
params = {
    'geo': '60763',
    'offset': '0',
    'establishmentTypes': '10591',
    'broadened': 'false'
}

restaurants = []

# Trip Advisor яростно сопротивляется парсингу с него, поэтому добавлены опции для имитации человеческого реального браузера
# source https://stackoverflow.com/questions/76611310/how-to-mimic-like-a-human-while-using-selenium-webdriver
def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--window-size=1920,1080')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(options=options)

    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver

def get_html(driver, url: str) -> str:

    try:
        driver.get(url)
        html = driver.page_source
        logging.info(f'Успешное обращение к страницу {url}')
        logging.debug(html)
    except Exception as e:
        logging.error(e)
        return ''

    return html

def parse_restaurants(html: str) -> None:
    soup = BeautifulSoup(html, 'html.parser')

    cards = soup.select('[data-automation="restaurantCard"]')
    for card in cards:
        is_sponsored = card.has_attr('data-sponsored')

        rating_div = card.select_one('div[data-automation="bubbleRatingValue"]')
        rating = "N/A"

        if rating_div:
            rating_span = rating_div.select_one('span')
            if rating_span:
                rating = rating_span.get_text(strip=True)

        review_count = 0
        review_count_div = card.select_one('div[data-automation="bubbleReviewCount"]')
        if review_count_div:
            review_count_span = review_count_div.select_one('span')
            if review_count_span:
                review_count_text = review_count_span.get_text(strip=True)
                numbers = re.findall(r'\d+', review_count_text)
                review_count = int(''.join(numbers))

        cuisine_type = ''
        details_span = card.select_one('span.f span')
        if details_span:
            cuisine_type = details_span.get_text(strip=True)

        price = ''
        price_span = card.select_one('div.biqBm > span.biGQs')
        if price_span:
            price = price_span.get_text(strip=True)

        restaurants.append({
            'rating': rating,
            'reviews_count': review_count,
            'cuisine_type': cuisine_type,
            'price': price,
            'is_sponsored': is_sponsored,
        })


def main():
    driver = get_driver()
    logging.info('Начало скрапинга')
    for i in range(0, 300):
        offset = i * 30
        current_params = params.copy()
        current_params['offset'] = str(offset)
        url = f'{BASE_URL}?{urlencode(current_params)}'

        html = get_html(driver, url)
        parse_restaurants(html)

        delay = random.uniform(2, 5) # имитация случайно задержки между переходами
        time.sleep(delay)
    logging.info('Скрапинг завершен')

    driver.quit()

    df = pd.DataFrame(restaurants)
    df.to_csv('./data/ta_restaurants.csv', index=False, encoding='utf-8')



if __name__ == '__main__':
    main()