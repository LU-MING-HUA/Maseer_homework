import requests
from bs4 import BeautifulSoup

web = requests.get("https://www.ntue.edu.tw/")
#print(web.text)
soup = BeautifulSoup(web.text, "html.parser")
print(soup)
