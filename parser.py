import os
import time
import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0"}
os.makedirs("data", exist_ok=True) 

courses = {}

print("--> Шаг 1: Ищем ссылки на 25 дисциплин...")
for page in [1, 2]:
    url = f"https://www.hse.ru/ba/isct/courses?course=1.1.1.4&page={page}&year=2026"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href", "")
        title = link.get_text().strip()

        if "/ba/isct/courses/" in href and href.endswith(".html") and not title.isdigit():
            if title not in courses:
                courses[title] = href

print(f"Найдено дисциплин: {len(courses)}")

print("\n--> Шаг 2: Скачиваем содержание каждой дисциплины...")
for i, (title, link) in enumerate(courses.items(), 1):
    print(f"[{i}/{len(courses)}] Скачиваем: {title}...")

    resp = requests.get(link, headers=headers)
    course_soup = BeautifulSoup(resp.text, "html.parser")

    content_div = course_soup.find("div", class_="post__text") or course_soup.find("div", class_="content")
    text = content_div.get_text(separator="\n", strip=True) if content_div else ""
    
    safe_name = title.replace('"', "").replace(":", "").replace("/", "_")
    filename = f"data/{i:02d}_{safe_name}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Дисциплина: {title}\n")
        f.write(f"Источник: {link}\n\n")
        f.write(text)

    time.sleep(0.3)  

print("\n Все дисциплины скачаны и сохранены в папку data/!")