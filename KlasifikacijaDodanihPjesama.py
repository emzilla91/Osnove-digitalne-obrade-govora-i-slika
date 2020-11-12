##%%

import Function
#import matplotlib.pyplot as plt
import pathlib
import librosa
import os
import csv
path="./TestYourSong/"
saveCSV="./datatest.csv"

header=Function.Header()
Function.OpeningCSV(header,saveCSV)
Function.WritingToCSV(path,saveCSV)


 
#%%
