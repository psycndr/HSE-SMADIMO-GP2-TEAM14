from bs4 import BeautifulSoup
import requests
import pandas as pd

BASE_URL = 'https://www.tripadvisor.ru/FindRestaurants'
params = {
    'geo': '60763',
    'offset': '0',
    'establishmentTypes': '10591',
    'broadened': 'false'
}