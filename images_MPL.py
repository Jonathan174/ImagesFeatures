# Load basic libraries.
from skimage.io import imread, imshow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Load and show an image in gray
image = imread('./Imagenes 2do proyecto/Vincent van Gogh/10.jpg')

# Check the image shape 
###print(image.shape) 
# Watch the content of the image in grayscale
###print(image)

size = image.shape[0]*image.shape[1]
feature_matrix_image = np.zeros((image.shape[0],image.shape[1]))
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        feature_matrix_image[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)

###print(feature_matrix_image)

feature_sample = np.reshape(feature_matrix_image, size) 
###print(feature_sample)
###print(feature_sample.shape)

column_names = ["Feature sample"]
data_df = pd.DataFrame(feature_sample, columns=column_names)
data_df.to_csv("TEST.csv", sep='\t', encoding='utf-8')
#print("\n\nThe DataFrame generated from the NumPy array is:")
#print(data_df)

# Watch the content of the image in grayscale
imshow(image)
plt.show()