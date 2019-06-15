

import numpy as np
import os
from tensorflow import keras
from PIL import Image

dimen = 128

dir_path = input( 'Enter images directory path : ')
out_path = input( 'Enter images output path : ')

sub_dir_list = os.listdir( dir_path )
images = list()
labels = list()
for i in range( len( sub_dir_list ) ):
	label = i
	image_names = os.listdir( os.path.join(dir_path , sub_dir_list[i]) )
	for image_path in image_names:
		path = os.path.join(dir_path , sub_dir_list[i] , image_path )
		try :
			image = Image.open(path)
			resize_image = image.resize((dimen, dimen))
			array_ = list()
			for x in range(dimen):
				sub_array = list()
				for y in range(dimen):
					sub_array.append(resize_image.load()[x, y])
				array_.append(sub_array)
			image_data = np.array(array_)
			image = np.array(np.reshape(image_data, (dimen, dimen, 3))) / 255
			images.append(image)
			labels.append(label)
		except:
			print( 'WARNING : File {} could not be processed.'.format( path ) )

images = np.array( images )

samples_1 = list()
samples_2 = list()
labels = list()
for i in range( 6 ) :
	for j in range( 6 ) :
		samples_1.append( images[i] )
		samples_2.append( images[j] )
		if i < 3 :
			if j < 3 :
				labels.append( 1 )
			else:
				labels.append( 0 )
		else :
			if j > 2 :
				labels.append( 1 )
			else:
				labels.append( 0 )

X1 = np.array( samples_1  )
X2 = np.array( samples_2 )
Y = np.array( labels )

np.save( '{}/x1.npy'.format( out_path ), X1 )
np.save( '{}/x2.npy'.format( out_path ), X2 )
np.save( '{}/y.npy'.format( out_path ) , Y )





