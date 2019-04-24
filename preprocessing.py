from python_speech_features import base
import os
import numpy as np
import pickle as pkl
import soundfile as sf

TEST_SPLIT = 0.2

# create necessary directories
if not os.path.exists('tools'):
    os.makedirs('tools')
if not os.path.exists('data'):
    os.makedirs('data')

training_data = []
for file in os.listdir("LibriSpeech"):
    if file.endswith(".txt"):
        lines = list(open(os.path.join("LibriSpeech", file)))
        for line in lines:
            separated = line.split(' ', 1)
            file_name = 'LibriSpeech/' + separated[0] + '.flac'
            transcript = separated[1]
            print(transcript)
            signal, fs = sf.read(file_name)

            mfcc_features = base.mfcc(signal, fs, 0.025, 0.01, 13,
                 20, 512, 0, None, 0.97, 22, True)
            training_data.append((mfcc_features, transcript))

# store the feature vectors - list of N
# each element of shape (L*, F)
x = np.array([e[0] for e in training_data])
print(x.shape)

# save the data
np.save('data/training_data_preprocessed', x)

# log
print('Saved preprocessed training data')

# store the labels (raw as strings)
with open('training_labels_raw.txt', 'w') as f:
    f.write(''.join([e[1] for e in training_data]))

# log
print('Saved raw training labels')

# one hot encoding for the characters of the labels
# didn't find a library, so doing it manually (might take
# a while running it)

# list of all possible characters
alphabet = list(set(''.join([e[1] for e in training_data])))

# save the encoding and decoding map
enc_map = {e : i for i, e in enumerate(alphabet)}
dec_map = {i : e for i, e in enumerate(alphabet)}
# <eos> symbol
dec_map[len(alphabet)] = ''

# save the encoding and decoding maps
with open("tools/enc_map.pkl","wb") as f:
    pkl.dump(enc_map, f)
    # log
    print('Saved encoding map.')
with open("tools/dec_map.pkl","wb") as f:
    pkl.dump(dec_map, f)
    # log
    print('Saved decoding map.')

# list of lists (different lengths, integers)
y_list = [[enc_map[ch] for ch in list(e[1])] for e in training_data]
# append the <eos> symbol to each list
y_list = [e + str(len(alphabet)) for e in y_list]

# save the final preprocessed labels
y = np.array(y_list)
np.save('data/training_labels_preprocessed', y)

# log
print('Saved preprocessed training labels')

# zip the data and patterns into dataset
# x: N * (L*, F)
# y: N * L*
dataset = list(zip(x, y))
dataset = np.random.permutation(dataset)

# split into training and testing
test_size = int(TEST_SPLIT * len(dataset))
dataset_test = dataset[:test_size]
dataset_train = dataset[test_size:]
x_train = np.array([e[0] for e in dataset_train])
x_test = np.array([e[0] for e in dataset_test])
y_train = np.array([e[1] for e in dataset_train])
y_test = np.array([e[1] for e in dataset_test])

# save the split training and testing data
np.save('data/training_data_preprocessed', x_train)
np.save('data/testing_data_preprocessed', x_test)
np.save('data/training_labels_preprocessed', y_train)
np.save('data/testing_labels_preprocessed', y_test)
