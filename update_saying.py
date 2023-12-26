import requests


hr_list=requests.get("https://the-brotherhood-of-scu.github.io/herui_saying_text/").text
with open("saying.txt","w",encoding="utf-8") as f:
    f.write(hr_list)
