import librosa
import os
from PIL import Image
import csv
import numpy as np
import warnings

#Extracting features from Spectrogram----------------------------------------------------
def Header():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    return header

#Saving data into csv--------------------------------------------------------------------------------
def OpeningCSV(header,saveCSV):
    file = open(f'{saveCSV}', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    return 0

#Writing features to csv for each song-------------------------------------------------------------------
def WritingToCSV(path,saveCSV):
    genres=[]
    for foldername in os.listdir(f'{path}'):
        genres.append(f'{foldername}')
    for g in genres:
        for filename in os.listdir(f'{path}{g}'):
            songname = f'{path}{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rmse(y=y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open(f'{saveCSV}', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

    return 0