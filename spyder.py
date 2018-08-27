import requests
from bs4 import BeautifulSoup
import os
from time import sleep

requests.adapters.DEFAULT_RETRIES = 5 # 解决”Max retries exceeded with url“的问题
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
           'Accept-Language': 'zh-CN,zh;q=0.9',
           'Connection': 'Close',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
path = "./figures/"
page_num = 0


def get_html(x):
    global page_num
    page_url = f"http://www.animecharactersdatabase.com/ux_search.php?x={x}&mimikko=0&tag=&gender=1&hair_color=0&hair_color2=0&hair_color3=0&hair_length=0&hair_length2=0&hair_length3=0&eye_color=0&eye_color2=0&eye_color3=0&age2=0&age3=0&age=0"
    r = requests.get(
        page_url,
        headers=headers)
    print(f"page{page_num} status code:{r.status_code}")
    page_num += 1
    return r.text


def get_img_urls(text):
    soup = BeautifulSoup(text, "html.parser")
    imgs = soup.find_all("img")
    img_urls = []
    for img in imgs:
        try:
            alt = img["alt"]
        except KeyError:
            continue
        if "uploads" in alt:
            img_urls.append(img["src"])

    return img_urls


def makefile(path):
    if not os._exists(path):
        try:
            os.makedirs(path)
        except:
            pass


def download_imgs(img_urls, x):
    global index
    for url in img_urls:
        im = requests.get(url, headers=headers)
        sleep(1)
        print(f"x={x},status code{im.status_code},fig_num{index}")
        with open(path + str(index) + ".jpg", 'wb') as f:
            f.write(im.content)
            index += 1


index = 20914
x = 20970
makefile(path)
while x <= 35790:
    text = get_html(x)


    print(f"x={x}")
    sleep(1)

    img_urls = get_img_urls(text)
    download_imgs(img_urls, x)

    x += 30