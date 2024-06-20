import requests
from bs4 import BeautifulSoup

web = requests.get("https://www.ntue.edu.tw/")
#print(web.text)
soup = BeautifulSoup(web.text, "html.parser")
#print(soup)

title = soup.title
print(title)
print(title.string)

a_tags = soup.find_all('span', string="國立臺北教育大學公開澄清聲明")
# print(a_tags)
for tag in a_tags:
    print(tag.string)

