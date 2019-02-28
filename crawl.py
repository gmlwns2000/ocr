# -*- coding:utf-8 -*-

# In[ ]:
#  coding: utf-8

from google_images_download import google_images_download

from multiprocessing import Process
import math

def f(keywords, subprocCount, depth):
    genk = ""
    for i, word in enumerate(keywords):
        if i != 0:
            genk+=","
        genk += word
    print("get", keywords)

    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": genk, "limit":int(2500), "print_urls":True, "chromedriver":"chromedriver.exe", "related_images":True}   #creating list of arguments

    absolute_image_paths = response.download(arguments)
    print(absolute_image_paths)

if __name__ == '__main__':
    keywords = ["console interface", "css text", "css font", "advertisement", "AD text", "apple typography", "cool text", "fancy text", "great text", "text design", "text design photoshop", "font design", "style text", "text ad", "poster","book cover","texts", "photoshop text", "glype", "fonts", "logo", "game logo", "photoshop text layer", "infographic", "powerpoint text", "handwriting", "handwriting alphabet", "fonts alphabet", "typography", "modern typography", "neet typography", "cool typography", "sans serif", "sans serif font", "serif", "serif font", "sans font", "cool sans font", "helvetica", "noto sans", "sans fonts", "noto sans fonts", "sans serif fonts", "serif fonts", "google fonts", "roboto", "roboto font", "roboto fonts", "ubuntu font", "ubuntu fonts", "open sans", "opensans font", "why fonts matter", "headline news", "photoshop font effect", "photoshop text effect", "gimp font effect", "gimp text effect", "photoshop font effects", "photoshop text effects", "gimp font effects", "gimp text effects", "illustrator cs6 font effect", "illustrator cs5 font effect", "illustrator cs3 font effect", "illustrator cs2 font effect", "newspaper front page", "newspaper headline", "adobe fonts", "adobe font family", "adobe font", "현수막 디자인", "CSS 폰트", "CSS 텍스트 스타일", "콘솔 인터페이스", "텍스트 디자인 포토샵", "포토샵 택스트 이팩트", "텍스트 디자인", "폰트 디자인", "스타일 텍스트", "글자 디자인", "광고", "텍스트 광고", "현수막", "포스터", "책표지", "글", "포토샵 택스트", "글자", "폰트", "로고", "게임 로고", "포토샵 텍스트 레이어", "인포그래픽", "파워포인트 택스트", "손글씨", "손글씨 글귀", "폰트 철자", "멋진 글자", "이쁜 글자", "이쁜 폰트", "텍스트 로고", "타이포그래피", "자막", "멋진 타이포 그래피", "모던 타이포그래피"]

    procs = []
    procsCount = 64

    for i in range(procsCount):
        k = []
        for _ in range(math.ceil(len(keywords) / procsCount)):
            if len(keywords) > 0:
                k.append(keywords.pop())
        if len(k) > 0:
            p = Process(target=f, args=(k, procsCount, 3))
            p.start()
            procs.append(p)

    for i in procs:
        i.join()