from matplotlib import pyplot as plt
from PIL import Image
import glob
import string

images = []

for letter in string.ascii_lowercase:
    current_images = []
    for filename in glob.glob(f'out/{letter}/*.png'):
        im=Image.open(filename)
        current_images.append(im)
        if len(current_images) == 10:
            break

    images.append(current_images)


fig, axs = plt.subplots(26,10, figsize=(10, 26))

for i in range(26):
 for j in range(10):
     axs[i,j].imshow(images[i][j], cmap=plt.cm.gray)
     axs[i,j].set_axis_off()

plt.savefig(f'dcgan_letters.pdf', format='pdf', bbox_inches='tight')
