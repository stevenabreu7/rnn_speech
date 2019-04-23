from python_speech_features import base
import os
import numpy as np
import soundfile as sf


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

signal, fs = sf.read('LibriSpeech/84-121123-0000.flac')
mfcc_features = base.mfcc(signal, fs, 0.025, 0.01, 13,
                 20, 512, 0, None, 0.97, 22, True)
print(len(mfcc_features))


# store the feature vectors - shape (20, N)
x = np.array([e[0] for e in training_data])
print(x.shape)

# store the data
np.save('training_data_preprocessed', x)

# store the labels (as strings
with open('training_labels_raw.txt', 'w') as f:
    f.write(''.join([e[1] for e in training_data]))
