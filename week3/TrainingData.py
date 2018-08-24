from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

class TrainingData(object):
	def __init__(self):
		self.destFemaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/female/'
		self.destMaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/male/'
		self.testFemaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/testing_female/'
		self.testMaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/testing_male/'
		pass

	def createModel(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(200,200,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		 
		return model

	def trainingModel(self):
		model = self.createModel()
		batch_size, epochs = 16, 100
		model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
		train_data_dir = 'training/male'
		validation_data_dir = 'training/testing_male'
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)
		test_datagen = ImageDataGenerator(rescale=1. / 255)

		train_generator = train_datagen.flow_from_directory(
			train_data_dir,  # this is the target directory
			target_size=(200, 200),  # all images will be resized to 150x150
			batch_size=batch_size,
			class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
		validation_generator = test_datagen.flow_from_directory(
			validation_data_dir,
			target_size=(200, 200),
			batch_size=batch_size,
			class_mode='binary')

		model.fit_generator(
				train_generator,
				steps_per_epoch=3565 // batch_size,
				epochs=epochs,
				validation_data=validation_generator,
				validation_steps=452 // batch_size)
		model.save_weights('first_try.h5')

if __name__ == '__main__':
	trainingData = TrainingData()
	trainingData.trainingModel()