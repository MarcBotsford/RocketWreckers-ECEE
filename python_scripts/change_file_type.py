import os
from PIL import Image
execution_path = os.getcwd()
directory = os.fsencode(execution_path)
total = 0
print(os.listdir(directory))
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".asm") or filename.endswith(".py") or filename.endswith(".jpg") : 
        # print(os.path.join(directory, filename))
        continue
    else:
        im = Image.open(filename)
        rgb_im = im.convert('RGB')
        rgb_im.save("plane_test_image_{}.jpg".format(
                str(total).zfill(8)))
        total = total + 1