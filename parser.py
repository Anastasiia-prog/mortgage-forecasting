import requests
from bs4 import BeautifulSoup
import csv
import time

CSV = 'flats.csv'
url = 'https://realty.yandex.ru/sankt-peterburg/kupit/kvartira/' # url страницы
HEADERS = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'}

def get_html(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    return r

def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('li', class_='OffersSerpItem')

    flats = []
    for item in items:
        flats.append({
            'date': item.find('div', class_='OffersSerpItem__publish-date').get_text(strip=True),
            'price': item.find('div', class_='OffersSerpItem__price-detail').get_text()
        })
    print(flats)

def save_doc(items, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['date', 'price'])
        for item in items:
            writer.writerow([item['date'], item['price']])


def parser():
    # pagenation = input('Введите количество страниц для парсинга ')
    # pagenation = int(pagenation.strip())
    pagenation = 20

    html = get_html(url)
    if html.status_code == 200:
        # get_content(html.text)
        flats = []
        for page in range(1, pagenation):
            print(f'Парсим страницу: {page}')
            html = get_html(url, params={'page': page})
            time.sleep(5)
            flats.extend(get_content(html.text))
            save_doc(flats, CSV)
            time.sleep(3)
        print(flats)

    else:
        print('Error')

parser()