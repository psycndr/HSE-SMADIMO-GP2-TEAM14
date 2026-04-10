import re

from bs4 import BeautifulSoup
import pandas as pd
import asyncio
from urllib.parse import urlencode

BASE_URL = 'https://www.tripadvisor.ru/FindRestaurants'
params = {
    'geo': '60763',
    'offset': '0',
    'establishmentTypes': '10591',
    'broadened': 'false'
}

async def get_html(page, offset) -> str:
    current_params = params.copy()
    current_params['offset'] = str(offset)
    url = f"{BASE_URL}?{urlencode(current_params)}"

    await page.goto(url, wait_until='networkidle', timeout=60000)

    html = await page.content()

    return html

def parse_restaurants(html):
    soup = BeautifulSoup(html, 'html.parser')
    restaurants = []

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
    with open('bbb.html', 'r', encoding='utf-8') as f:
        parse_restaurants(f.read())



if __name__ == '__main__':
    main()