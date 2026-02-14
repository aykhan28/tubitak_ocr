import os

a = os.listdir("yonetim_output")

for i in a:
    os.system(f'python main_evaluate.py "yonetim_output/{i}" "dogru_cevaplar.json"')