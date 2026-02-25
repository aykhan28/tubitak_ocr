import os

a = os.listdir("puan")

for i in a:
    os.system(f'python main_puan.py "puan/{i}"')