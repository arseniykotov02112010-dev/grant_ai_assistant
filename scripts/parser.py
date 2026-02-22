import requests
from bs4 import BeautifulSoup
import os


def download_file(url, папка='/Users/macbook/PycharmProjects/PythonProject/parser/docs', имя_файла=None):

    try:
        # Создаем папку если её нет
        os.makedirs(папка, exist_ok=True)

        # Получаем имя файла из URL если не указано
        if имя_файла is None:
            имя_файла = url.split('/')[-1]

        # Полный путь для сохранения
        путь_файла = os.path.join(папка, имя_файла)

        # Скачиваем файл
        print(f"Скачиваю: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверяем на ошибки

        # Сохраняем файл
        with open(путь_файла, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"Файл сохранен: {путь_файла}")
        return путь_файла

    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return None


for i in range(1, 19):
    url = f'https://rscf.ru/contests/?PAGEN_2={i}'

    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'lxml')

    container = soup.select_one('div[id^="comp_"]')

    data = container.find('div', class_="classification-table mb-0")

    st = data.find_all('div', class_='classification-table-row classification-parent-row contest-table-row')

    for j in st:
        a = j.find_all('a', class_="contest-link")[1]

        pdf = 'https://rscf.ru' + a.get('href')
        download_file(pdf)
        print(pdf)