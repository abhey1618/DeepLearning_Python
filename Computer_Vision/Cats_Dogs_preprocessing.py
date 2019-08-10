import os, shutil

##Directory which contains the images
original_dir = 'F:/main_abhey/M1/Extra_Courses/Kaggle/dogs-vs-cats/train'

##Directory that will contain the sub folders
base_dir = 'F:/main_abhey/M1/Extra_Courses/Kaggle/dogs-vs-cats_sample'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

##Making a 60-20-20 split.

fnames = ['cat.{}.jpg'.format(i) for i in range(7500)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(train_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(7500, 10000)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(validation_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(10000, 12500)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(test_cats_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(7500)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(train_dogs_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(7500, 10000)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(validation_dogs_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(10000, 12500)]
for fname in fnames:
	src = os.path.join(original_dir, fname)
	dst = os.path.join(test_dogs_dir, fname)
	shutil.copyfile(src, dst)
