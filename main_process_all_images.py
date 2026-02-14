import os

a = os.listdir("projeyonetimi")

for i in a:
    os.system(f'python main_v3.py "projeyonetimi/{i}"')