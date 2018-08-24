import csv, cv2, os

class SeperateImage(object):
	def __init__(self):
		self.srcAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/image/'
		self.publicData = '/Users/minhvu/workplace/machineLearning/trainingdata/public-test-data/image/'
		self.destFemaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/female/'
		self.destMaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/male/'
		self.testFemaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/female/'
		self.testMaleAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/male/'
		pass

	def readingCSV(self, fileName):
		count = 0
		with open(fileName) as f:
			reader = csv.DictReader(f)
			for row in reader:
				imageFileName = self.srcAddress+row['userid']+'.jpg'
				image = cv2.imread(imageFileName)
				newImage = cv2.resize(image, (200, 200)) 
				if count >= 8500:
					if row['gender'] == '0.0':
						cv2.imwrite(os.path.join(self.testMaleAddress, row['userid']+'.jpg'), newImage)
					else:
						cv2.imwrite(os.path.join(self.testFemaleAddress, row['userid']+'.jpg'), newImage)
					count += 1
				else:
					if row['gender'] == '0.0':
						cv2.imwrite(os.path.join(self.destMaleAddress, row['userid']+'.jpg'), newImage)
					else:
						cv2.imwrite(os.path.join(self.destFemaleAddress, row['userid']+'.jpg'), newImage)
					count += 1


class SeperateImageByAge(object):
	def __init__(self):
		self.srcAddress = '/Users/minhvu/workplace/machineLearning/trainingdata/training/image/'
		self.destAge24 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/age/xx-24'
		self.destAge25 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/age/25-34'
		self.destAge35 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/age/35-49'
		self.destAge50 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/trainingdata/age/50-xx'
		self.testAge24 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/age/xx-24'
		self.testAge25 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/age/25-34'
		self.testAge35 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/age/35-49'
		self.testAge50 = '/Users/minhvu/workplace/machineLearning/trainingdata/training/validation/age/50-xx'

	def readingCSV(self, fileName):
		count = 0
		with open(fileName) as f:
			reader = csv.DictReader(f)
			for row in reader:
				imageFileName = self.srcAddress+row['userid']+'.jpg'
				age = int(float(row['age'])) 
				image = cv2.imread(imageFileName)
				newImage = cv2.resize(image, (200, 200)) 
				if count >= 8500:
					if age <= 24:
						cv2.imwrite(os.path.join(self.testAge24, row['userid']+'.jpg'), newImage)
					elif age >= 25 and age <= 34:
						cv2.imwrite(os.path.join(self.testAge25, row['userid']+'.jpg'), newImage)
					elif age >=35 and age <= 49:
						cv2.imwrite(os.path.join(self.testAge35, row['userid']+'.jpg'), newImage)
					elif age >=50:
						cv2.imwrite(os.path.join(self.testAge50, row['userid']+'.jpg'), newImage)
				else:
					if age <= 24:
						cv2.imwrite(os.path.join(self.destAge24, row['userid']+'.jpg'), newImage)
					elif age >= 25 and age <= 34:
						cv2.imwrite(os.path.join(self.destAge25, row['userid']+'.jpg'), newImage)
					elif age >=35 and age <= 49:
						cv2.imwrite(os.path.join(self.destAge35, row['userid']+'.jpg'), newImage)
					elif age >=50:
						cv2.imwrite(os.path.join(self.destAge50, row['userid']+'.jpg'), newImage)
				count += 1


if __name__ == '__main__':
	# seperateImage = SeperateImage()
	seperateImage = SeperateImageByAge()
	seperateImage.readingCSV('../trainingdata/training/profile/profile.csv')
	
