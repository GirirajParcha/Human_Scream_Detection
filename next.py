import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow as tf 
import librosa

scream = "scream/1.wav"
non_scream = "non_scream/1.wav"

def load_wav_16k_mono(filename):
    # Load the audio file with librosa
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    # Convert to tensorflow tensor
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    return wav

num = 1011
scream = f"scream/{num}.wav"
non_scream = f"non_scream/{num}.wav"

wave = load_wav_16k_mono(scream)
nwave = load_wav_16k_mono(non_scream)

plt.plot(wave)
plt.plot(nwave)
plt.show()

POS = "/scream"
NEG = "non_scream"

pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

negatives.as_numpy_iterator().next()

lengths = []
for file in os.listdir("/kaggle/input/audio-dataset-of-scream-and-non-scream/Converted_Separately/scream"):
    tensor_wave = load_wav_16k_mono(f"/kaggle/input/audio-dataset-of-scream-and-non-scream/Converted_Separately/scream/{file}")
    lengths.append(len(tensor_wave))

    np.mean(lengths)
    lengths[np.argmin(lengths)]
    lengths[np.argmax(lengths)]
    np.std(lengths)

    def preprocess(file_path, label): 
        wav = load_wav_16k_mono(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram, label
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
filepath, label
spectrogram, label = preprocess(filepath, label)

plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

#Her seferinde codu çalıştırarak farklı bir sesin spectrogramını görebilirsiniz

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next() #scream sounds
#filepath, label = negatives.shuffle(buffer_size=10000).as_numpy_iterator().next() #non scream sounds
spectrogram, label = preprocess(filepath, label)

plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

x = data
data = x
len(data)

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=3128)  # Shuffle the entire dataset
data = data.batch(16)  # Adjust batch size depending on memory
data = data.prefetch(tf.data.AUTOTUNE)  # Use AUTOTUNE for dynamic prefetching

train = data.take(137)  # Take 137 samples for training (70%)
val = data.skip(137).take(19)  # Skip training samples and take 19 for validation (10%)
test = data.skip(137 + 19).take(39)  # Take remaining 39 samples for testing (20%)

samples, labels = train.as_numpy_iterator().next()
samples.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# First Conv Layer with Pooling
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduce dimensions by half

# Second Conv Layer with Pooling
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Further reduce dimensions by half

# Flatten the reduced output
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='Adam', 
              loss='BinaryCrossentropy', 
              metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.summary()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# First Conv Layer with Pooling
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduce dimensions by half

# Second Conv Layer with Pooling
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Further reduce dimensions by half

# Flatten the reduced output
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='Adam', 
              loss='BinaryCrossentropy', 
              metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.summary()

hist = model.fit(train, epochs=4, validation_data=val)

hist.history

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision_1'], 'r')
plt.plot(hist.history['val_precision_1'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall_1'], 'r')
plt.plot(hist.history['val_recall_1'], 'b')
plt.show()

X_test, y_test = test.as_numpy_iterator().next()
X_test.shape
y_test.shape
yhat = model.predict(X_test)

yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, yhat)
print(classification_report(y_test, yhat))

model.save('my_model.h5')  # Saves the model as an HDF5 file

from tensorflow.keras.models import load_model

model = load_model('my_model.h5')