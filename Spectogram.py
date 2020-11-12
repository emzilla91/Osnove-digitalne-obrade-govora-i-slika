#%%
#spremanje spectograma u datoteku--------------------------------------------------------
import matplotlib.pyplot as plt
import pathlib
import librosa
import os

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'/run/media/gabriel/HDD/Projekt/klasifikacija-glazbenih-zanrova/TestYourSong/{g}'):
            songname = f'/run/media/gabriel/HDD/Projekt/klasifikacija-glazbenih-zanrova/TestYourSong/{g}/{filename}'
            y, sr = librosa.load(songname, duration=5.0)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('off')
            plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()

