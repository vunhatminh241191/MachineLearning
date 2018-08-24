import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras import applications
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
from keras import backend as K
K.set_image_dim_ordering('th')

class DataWithBottleNeck(object):
	def __init__(self):
		self.top_model_weights_path = 'bottleneck_fc_model.h5'
		self.train_data_dir = 'trainingdata'
		self.validation_data_dir = 'validation'
		self.nb_train_samples = 8500
		self.nb_validation_samples = 1000
		self.epochs = 50
		self.batch_size = 40
		self.testingModelFolder = 'image'
		self.publicData = '../public-test-data/image/'

	def save_bottlebeck_features(self):
		datagen = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True
			)

		# build the VGG16 network
		model = applications.VGG_16('vgg16_weights.h5')

		generator = datagen.flow_from_directory(
			self.train_data_dir,
			target_size=(200, 200),
			batch_size=self.batch_size,
			class_mode=None,
			shuffle=False)
		bottleneck_features_train = model.predict_generator(
			generator, self.nb_train_samples // self.batch_size)
		np.save(open('bottleneck_features_train.npy', 'w'),
				bottleneck_features_train)

		generator = datagen.flow_from_directory(
			self.validation_data_dir,
			target_size=(200, 200),
			batch_size=self.batch_size,
			class_mode=None,
			shuffle=False)
		bottleneck_features_validation = model.predict_generator(
			generator, self.nb_validation_samples // self.batch_size)
		np.save(open('bottleneck_features_validation.npy', 'w'),
				bottleneck_features_validation)


	def train_top_model(self):
		train_data = np.load(open('bottleneck_features_train.npy'))

		# 4935 is total image of female label
		# 3565 is total image of male label
		train_labels = np.array([0] * 4935 + [1] * 3565)

		# 548 is total image of female label in testing
		# 452 is total image of male label in testing
		validation_data = np.load(open('bottleneck_features_validation.npy'))
		validation_labels = np.array([0] * 548 + [1] * 452)

		model = self.VGG_16('vgg16_weights.h5')
		print mode.output_shape
		model = Sequential()
		model.add(Flatten(input_shape=train_data.shape[1:]))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))

		model.compile(optimizer='rmsprop',
					  loss='binary_crossentropy', metrics=['accuracy'])
		checkpoint = ModelCheckpoint(self.top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True
			, save_weights_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(train_data, train_labels,
				  epochs=self.epochs,
				  batch_size=self.batch_size,
				  validation_data=(validation_data, validation_labels),
				  callbacks=callbacks_list,
				  verbose=0
		)

		model_json = model.to_json()
		with open('model.json', 'w') as json_file:
			json_file.write(model_json)

		model.save('gender.h5')


	def predict_image_class(self):
		model = applications.VGG16(include_top=False, weights='imagenet')
		femaleCount, maleCount = 0, 0

		for fileName in os.listdir(self.publicData):
			image = load_img(os.path.join(self.publicData + fileName), target_size=(200, 200))
			image = img_to_array(image)  

			# important! otherwise the predictions will be '0'  
			image = image / 255  

			image = np.expand_dims(image, axis=0) 
			bottleneck_prediction = model.predict(image) 
			top_model = Sequential()
			top_model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
			top_model.add(Dense(256, activation='relu'))
			top_model.add(Dropout(0.5))
			top_model.add(Dense(1, activation='sigmoid'))
			top_model.load_weights(self.top_model_weights_path) 

			class_predicted = top_model.predict_classes(bottleneck_prediction)
			if class_predicted[0][0] == 1:
				maleCount += 1
			elif class_predicted[0][0] == 0:
				femaleCount += 1
			print class_predicted, fileName
		print femaleCount, maleCount

solution = DataWithBottleNeck()
# solution.save_bottlebeck_features()
solution.train_top_model()
# solution.predict_image_class()