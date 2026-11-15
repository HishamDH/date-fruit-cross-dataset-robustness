"""Stack several review sheets vertically into one image. Usage: stack_sheets.py <out.jpg> <in1.jpg> <in2.jpg> ..."""
import sys

from PIL import Image, ImageDraw

ims = [Image.open(p) for p in sys.argv[2:]]
w = max(i.size[0] for i in ims)
out = Image.new('RGB', (w, sum(i.size[1] + 22 for i in ims)), (60, 0, 0))
dr = ImageDraw.Draw(out)
y = 0
for p, im in zip(sys.argv[2:], ims):
    dr.text((4, y + 4), p.split('_review_')[-1], fill=(255, 255, 255))
    out.paste(im, (0, y + 22))
    y += im.size[1] + 22
out.save(sys.argv[1], quality=80)
