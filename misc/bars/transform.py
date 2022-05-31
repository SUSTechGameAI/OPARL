"""
  @Time : 2022/2/18 20:34 
  @Author : Ziqi Wang
  @File : transform.py 
"""

from PIL import Image

if __name__ == '__main__':
    img = Image.open('backup/bardemo-div.png')
    img.save('./bar-div.png', dpi=(400, 400))
    img = Image.open('backup/bardemo-eps_all.png')
    img.save('./bar-eps_all.png', dpi=(400, 400))
    img = Image.open('backup/bardemo-fun.png')
    img.save('./bar-fun.png', dpi=(400, 400))
