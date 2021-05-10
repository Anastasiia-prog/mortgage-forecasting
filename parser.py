from selenium import webdriver
import time
import csv

class YandexParser(object):
    def __init__(self, driver, flat):
        self.driver = driver
        self.flat = flat

    def parse(self):
        self.go_to_flat_page()


    def go_to_flat_page(self):
        for num_page in range(25):
            self.driver.get(f'https://realty.yandex.ru/sankt-peterburg/kupit/kvartira/?page={num_page}')
            links = self.driver.find_elements_by_class_name('OffersSerpItem__link')
            flat_link_set = set()
            # flat_link = []
            print(num_page)

            for elem in links:
                # получить все ссылки со страницы и вывести их на экран
                flat_link_set.add(elem.get_attribute('href'))
                # flat_link.append(elem.get_attribute('href'))
                # print(flat_link)

            flat_page_date = dict()
            flat_link = list(flat_link_set)

            # зайти по ссылке
            for i in range(len(flat_link)):
                self.driver.get(flat_link[i])
                for_flat_page = self.driver.find_elements_by_css_selector("div[class='OfferPublishedDate OfferBaseMetaInfo__item']") # Собираем информацию о дате публикации
                time.sleep(2)

                test_fl = open('parse_link_date.csv', 'a', encoding='utf-8')

                with test_fl:
                    writer = csv.writer(test_fl)
                    for elem in for_flat_page:
                    # получить дату после перехода на страницу квартиры
                        flat_page_date[flat_link[i]] = elem.text.split(',')
                        writer.writerow(flat_page_date.popitem())
                        time.sleep(1.5)


def main():
    with webdriver.Chrome('chromedriver.exe') as driver:
        parser = YandexParser(driver, 'offer')
        parser.parse()

if __name__=='main':
    main()

main()