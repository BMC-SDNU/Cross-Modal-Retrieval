import os
import numpy as np
import librosa
import librosa.display
import scipy.io as scio
import matplotlib.pyplot as plt

pathDir = './dataset/audio/'
for classpath in os.listdir(pathDir):
    subPath = os.path.join(pathDir,classpath)
    newDir = './dataset/audioimg/' + classpath
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    for eachfile in os.listdir(subPath):
        filePath = os.path.join(subPath,eachfile)
        pngname = os.path.splitext(eachfile)[0] + '.png'
        saveData = os.path.join(newDir,classpath,pngname)
        try:
            y, sr = librosa.load(filePath)
            D = np.abs(librosa.stft(y))
            plt.figure()
            librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max))
            fig = plt.gcf()
            fig.set_size_inches(7.0/3,7.0/3)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig(saveData, format='png', transparent=True, dpi=300, pad_inches = 0) 
            plt.close()
        except:
            print(filePath+' read fail')

