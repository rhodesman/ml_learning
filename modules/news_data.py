import requests
from bs4 import BeautifulSoup

def fetch_news(query, max_articles=10):
    url = f"https://news.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("a", class_="DY5T1d", limit=max_articles)
    return [{"title": a.text, "link": "https://news.google.com" + a["href"][1:]} for a in articles]