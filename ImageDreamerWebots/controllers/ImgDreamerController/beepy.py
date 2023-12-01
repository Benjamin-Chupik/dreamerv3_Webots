import numpy as np
import matplotlib.pyplot as plt


img = np.load('/Users/guidoinsinger/Documents/GitHub/dreamerv3_webots/ImageDreamerWebots/controllers/ImgDreamerController/beepyview.npy')
plt.imshow(img)
plt.show()
print(img)