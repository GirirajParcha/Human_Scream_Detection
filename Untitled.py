#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install torchaudio


# In[3]:


#pip install torchaudio --index-url https://download.pytorch.org/whl/cu118


# In[4]:


import torchaudio
print(torchaudio.__version__)


# In[5]:


import os

import torchaudio
import torch
import matplotlib.pyplot as plt
from pathlib import Path


# In[6]:


def load_audio_files(path: str, label: str):
    # Initialize an empty list to store the dataset
    dataset = []

    # List all WAV files in the specified path
    walker = sorted(str(p) for p in Path(path).glob('*.wav'))

    # Iterate over the list of audio file paths
    for i, file_path in enumerate(walker):
        # Split the file path into directory and filename
        path, filename = os.path.split(file_path)
        name, _ = os.path.splitext(filename)

        # Load the audio waveform and sample rate using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        # Create an entry for the dataset, including waveform, sample rate, label, and ID
        entry = {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'id': i}

        # Append the entry to the dataset list
        dataset.append(entry)

    # Return the populated dataset
    return dataset
# Load audio files from the 'Data/Screaming' directory with the label 'yes'


# In[7]:


#pip install librosa


# In[8]:


#pip install soundfile audioread


# In[9]:

# Update the path to use absolute paths
screaming_dataset = load_audio_files('D:/Tskshm/scream', 'yes')

# Display the 'screaming_dataset' which contains loaded audio data
screaming_dataset[:5]


# In[10]:


not_screaming_dataset = load_audio_files('D:/Tskshm/non_scream', 'not')

# Display the 'screaming_dataset' which contains loaded audio data
not_screaming_dataset[:5]

# In[11]:


print(len(screaming_dataset), len(not_screaming_dataset))


# In[12]:


scream_id = 0  # Changed from 5 to 0
no_scream_id = 0

# Access the waveform and sample rate of the first entry in the 'screaming_dataset'
screaming_waveform = screaming_dataset[scream_id]['waveform']
screaming_sample_rate = screaming_dataset[scream_id]['sample_rate']

# Print the waveform, sample rate, label, and ID of the first entry
print(f'Screaming Waveform: {screaming_waveform}')
print(f'Screaming Sample Rate: {screaming_sample_rate}')
print(f'Screaming Label: {screaming_dataset[scream_id]["label"]}')
print(f'Screaming ID: {screaming_dataset[scream_id]["id"]} \n')

# Access the waveform and sample rate of the first entry in the 'screaming_dataset'
not_screaming_waveform = not_screaming_dataset[no_scream_id]['waveform']
not_screaming_sample_rate = not_screaming_dataset[no_scream_id]['sample_rate']

# Print the waveform, sample rate, label, and ID of the first entry
print(f'Screaming Waveform: {not_screaming_waveform}')
print(f'Screaming Sample Rate: {not_screaming_sample_rate}')
print(f'Screaming Label: {not_screaming_dataset[no_scream_id]["label"]}')
print(f'Screaming ID: {not_screaming_dataset[no_scream_id]["id"]} \n')


# In[13]:


def show_waveform(waveform, sample_rate, label):
    # Print information about the waveform, sample rate, and label
    print("Waveform: {}\nSample rate: {}\nLabels: {} \n".format(waveform, sample_rate, label))

    # Calculate the new sample rate by dividing the original sample rate by 10
    new_sample_rate = sample_rate / 10

    # Resample applies to a single channel, so we resample the first channel here
    channel = 0

    # Apply resampling to the waveform
    waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))

    # Print the shape of the transformed waveform and the new sample rate
    print("Shape of transformed waveform: {}\nSample rate: {}".format(waveform_transformed.size(), new_sample_rate))

    # Plot the transformed waveform
    plt.figure()
    plt.plot(waveform_transformed[0, :].numpy())


# In[14]:


show_waveform(screaming_waveform, screaming_sample_rate, 'yes')


# In[15]:


import IPython.display as ipd

# Plat the sample sound
ipd.Audio(screaming_waveform.numpy(), rate=screaming_sample_rate)


# In[16]:


def show_spectrogram(waveform_classA, waveform_classB):
    # Compute the spectrogram for the first waveform (class A)
    yes_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classA)
    print("\nShape of yes spectrogram: {}".format(yes_spectrogram.size()))

    # Compute the spectrogram for the second waveform (class B)
    no_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classB)
    print("Shape of no spectrogram: {}".format(no_spectrogram.size()))

    # Create a figure with two subplots for visualization
    plt.figure()

    # Plot the spectrogram of class A (left subplot)
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('yes'))

    # Set the aspect ratio to 'auto' for the y-axis to elongate it
    plt.imshow(yes_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto')

    # Plot the spectrogram of class B (right subplot)
    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('no'))

    # Set the aspect ratio to 'auto' for the y-axis to elongate it
    plt.imshow(no_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto')


# In[17]:


show_spectrogram(screaming_waveform, not_screaming_waveform)


# In[18]:


def show_mel_spectrogram(waveform, sample_rate):
    # Compute the Mel spectrogram from the input waveform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=1024
    )(waveform)

    # Print the shape of the Mel spectrogram
    print("Shape of spectrogram: {}".format(mel_spectrogram.size()))

    # Create a new figure for visualization
    plt.figure()

    # Display the Mel spectrogram as an image with a color map 'viridis'
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto')


# In[19]:


show_mel_spectrogram(screaming_waveform, screaming_sample_rate)


# In[20]:


show_mel_spectrogram(not_screaming_waveform, not_screaming_sample_rate)


# In[21]:


def show_mfcc(waveform, sample_rate):
    # Compute the MFCC spectrogram from the input waveform
    mfcc_spectrogram = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=128
    )(waveform)

    # Print the shape of the MFCC spectrogram
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    # Create a new figure for visualization
    plt.figure()

    # Display the MFCC spectrogram as an image with a color map 'viridis'
    plt.imshow(mfcc_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto')

    # Create a separate figure for the MFCC plot with an elongated y-axis
    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0, :, :].numpy())
    plt.draw()


# In[22]:


show_mfcc(screaming_waveform,  screaming_sample_rate)


# In[23]:


show_mfcc(not_screaming_waveform,  not_screaming_sample_rate)


# In[24]:


def pad_waveform(waveform, target_length):
    _, num_channels, current_length = waveform.shape

    if current_length < target_length:
        # Calculate the amount of padding needed
        padding = target_length - current_length
        # Pad the waveform with zeros on the right side
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform


def create_images(train_loader, label_dir, amplitude_threshold=0.01):
    # Make directory
    directory = f'Data/Images/{label_dir}/'
    if os.path.isdir(directory):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)

        for i, data in enumerate(train_loader):
            waveform = data['waveform']

            # Pad waveform to a consistent length of 44100 samples
            waveform = pad_waveform(waveform, 441000)

            # Check if the waveform has sufficient amplitude
            if torch.max(torch.abs(waveform)) > amplitude_threshold:
                # Create transformed waveforms
                spectrogram_tensor = torchaudio.transforms.MelSpectrogram(
                    sample_rate=int(data['sample_rate']),
                    n_mels=64,
                    n_fft=1024,
                )(waveform)

                plt.imsave(f'Data/Images/{label_dir}/audio_img{i}.png', (spectrogram_tensor[0] + 1e-10).log2()[0, :, :].numpy(), cmap='viridis')
            else:
                print(f'Skipping blank waveform {i} in {label_dir}')


# In[25]:


from torch.utils.data import DataLoader

train_loader_scream = DataLoader(screaming_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
train_loader_not_scream = DataLoader(not_screaming_dataset, batch_size=1,
                                             shuffle=False, num_workers=0)


# In[26]:


create_images(train_loader_scream, 'scream')


# In[27]:


create_images(train_loader_not_scream, 'not')


# In[28]:


from torchvision.transforms import Lambda
import torch

def random_time_shift(audio, max_shift_ms=1000, target_length=441000):
    # Calculate the current length of the audio
    current_length = audio.shape[-1]

    # If the audio is longer than the target length, perform random shift
    if current_length >= target_length:
        shift = torch.randint(-max_shift_ms, max_shift_ms, (1,)).item()
        shift_samples = int(shift * 44100 / 1000)  # Assuming a 44.1 kHz sample rate
        if shift_samples >= 0:
            audio = torch.nn.functional.pad(audio, (shift_samples, 0))
            # Cut the audio from the end to fit the target_length
            audio = audio[:, :target_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, -shift_samples))
            # Cut the audio from the front to fit the target_length
            audio = audio[:, -target_length:]
    else:
        # If the audio is shorter than the target length, pad it on both ends
        padding = target_length - current_length
        # Calculate shift_samples to be greater than 0
        shift_samples = torch.randint(1, padding, (1,)).item()

        # Determine the left padding as a random number within the range [0, shift_samples]
        left_padding = torch.randint(0, shift_samples, (1,)).item()
        
        # Calculate the right padding as the difference between padding and left_padding
        right_padding = padding - left_padding

        audio = torch.nn.functional.pad(audio, (left_padding, right_padding))

    return audio


# In[29]:


def add_noise(audio, noise_level=0.005):
    noise = noise_level * torch.randn_like(audio)
    noisy_audio = audio + noise
    return noisy_audio


# In[30]:


from torchvision.transforms import Compose

def augment_audio(audio):
    transform_audio = Compose([
        Lambda(lambda x: random_time_shift(x))
    ])

    augmented_audio = transform_audio(audio)

    return augmented_audio


# In[31]:


shift_audio = augment_audio(screaming_waveform)
shift_audio.shape


# In[32]:


show_waveform(shift_audio, screaming_sample_rate, 'Test')


# In[33]:


def create_shift_images(train_loader, label_dir, amplitude_threshold=0.01, shift_time=0):
    for i, data in enumerate(train_loader):
        waveform = data['waveform']

        # Generate shifting audio
        waveform = random_time_shift(waveform)

        # Check if the waveform has sufficient amplitude
        if torch.max(torch.abs(waveform)) > amplitude_threshold:
            # Create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.MelSpectrogram(
                sample_rate=int(data['sample_rate']),
                n_mels=64,
                n_fft=1024,
            )(waveform)

            if spectrogram_tensor[0].log2().isnan().any() or spectrogram_tensor[0].log2().isinf().any():
                continue

            plt.imsave(f'Data/Images/{label_dir}/audio_img{i}_shift{shift_time}.png', spectrogram_tensor[0].log2()[0, :, :].numpy(), cmap='viridis')
        else:
            print(f'Skipping blank waveform {i} in {label_dir}')

    return 'Done!'


# In[34]:


# Call the function three times with different shift_time values using list comprehension
[create_shift_images(train_loader_scream, 'scream', shift_time=i) for i in range(5)]


# In[ ]:





import matplotlib.pyplot as plt

def visualize_all_audio_features(waveform_scream, waveform_non_scream, sample_rate):
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Waveform Comparison
    plt.subplot(3, 2, 1)
    plt.title('Scream Waveform')
    plt.plot(waveform_scream[0].numpy())
    
    plt.subplot(3, 2, 2)
    plt.title('Non-Scream Waveform')
    plt.plot(waveform_non_scream[0].numpy())
    
    # 2. Spectrograms
    spectrogram_scream = torchaudio.transforms.Spectrogram()(waveform_scream)
    spectrogram_non_scream = torchaudio.transforms.Spectrogram()(waveform_non_scream)
    
    plt.subplot(3, 2, 3)
    plt.title('Scream Spectrogram')
    plt.imshow(spectrogram_scream.log2()[0, :, :].numpy(), aspect='auto', cmap='viridis')
    
    plt.subplot(3, 2, 4)
    plt.title('Non-Scream Spectrogram')
    plt.imshow(spectrogram_non_scream.log2()[0, :, :].numpy(), aspect='auto', cmap='viridis')
    
    # 3. Mel Spectrograms
    mel_scream = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform_scream)
    mel_non_scream = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform_non_scream)
    
    plt.subplot(3, 2, 5)
    plt.title('Scream Mel Spectrogram')
    plt.imshow(mel_scream.log2()[0, :, :].numpy(), aspect='auto', cmap='viridis')
    
    plt.subplot(3, 2, 6)
    plt.title('Non-Scream Mel Spectrogram')
    plt.imshow(mel_non_scream.log2()[0, :, :].numpy(), aspect='auto', cmap='viridis')
    
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_all_audio_features(screaming_waveform, not_screaming_waveform, screaming_sample_rate)
