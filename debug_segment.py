import numpy as np
import skimage.io as io

tmp = np.transpose((img[0].detach().cpu().numpy() + 0.5) * 255, (1, 2, 0))
io.imsave('temp.jpg', tmp)