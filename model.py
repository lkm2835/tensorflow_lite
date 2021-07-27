import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#train_images, test_images = train_images / 255.0, test_images / 255.0

#test = np.array([train_images[0], train_images[0]])
#test = test.reshape(1,28,28,2)
#print(test.shape)

new_images = list()
#new_test_images = list()

for i in range(60000):
	temp = np.concatenate((train_images[i], train_images[i]), axis=1)
	new_images.append(np.concatenate((temp, temp), axis=0))

new_train_images = np.array(new_images, dtype=np.float32)
new_images.clear()

for i in range(10000):
	temp = np.concatenate((test_images[i], test_images[i]), axis=1)
	new_images.append(np.concatenate((temp, temp), axis=0))

new_test_images = np.array(new_images, dtype=np.float32)

new_train_imagesm, new_test_images = new_train_images / 255.0, new_test_images / 255.0



print(train_images.shape)
print(test_images.shape)
print(new_train_images.shape)
print(new_test_images.shape)



#test = np.concatenate((train_images[0], train_images[0]), axis=1)
#test = np.concatenate((test, test), axis=0)
#print(test.shape)

#for k in range(2):
"""
for k in range(test.shape[2]):
	for i in range(test.shape[0]):
		for j in range(test.shape[1]):
			print(test[i][j][k], end='\t')
		print()
	print()
"""
"""
train_images, test_images = train_images / 255.0, test_images / 255.0

print()
print()
print()
print()


#############################
model1 = models.Sequential()
model1.add(layers.Conv2D(2, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.Conv2D(1, (3, 3), activation='relu'))
#model1.add(layers.MaxPooling2D((2, 2)))
#model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model1.add(layers.MaxPooling2D((2, 2)))
#model1.add(layers.Conv2D(60, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))


model1.summary()

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model1.fit(train_images, train_labels, epochs=5)

############################

###

model1.evaluate(test_images, test_labels, verbose=0)

tf.saved_model.save(model1, "./model/")


saved_model_dir = "./model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

with open('converted_model_conv_28x28x1_2_1.tflite', 'wb') as f:
	f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="converted_model_conv_28x28x1_2_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print()
print(output_details)

print(input_details[0]['quantization_parameters'])

acc = 0
for num in range(10):
	input_shape = input_details[0]['shape']
	input_data = np.array([test_images[num]], dtype=np.float32)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	interpreter.invoke()

	output_data = interpreter.get_tensor(output_details[0]['index'])
	for i in range(28):
		for j in range(28):
			print(input_data[0][i][j], end='\t')
		print()
	print(test_labels[num])
	print(np.argmax(output_data))
	if test_labels[num] == np.argmax(output_data):
		acc += 1

print(acc / 10)
"""

print("TEST")
saved_path = './model/'
loaded = tf.keras.models.load_model(saved_path)

weight = loaded.get_weights()
print("TEST")
#print(weight[2][0][0])
print()
#print(weight[2][1][0])
print()
#print(weight[2][2][0])
print(weight[1])
print()
print(weight[3])
