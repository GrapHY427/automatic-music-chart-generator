import torch
import torchaudio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from create_chart import *


class DatasetGenerator:

    def __init__(self, path):
        self.path = path + '.wav'
        self.chart_generator = ChartGenerator(path)
        self.spectrogram = self.spectrogram_transformer()

    def spectrogram_transformer(self):
        waveform, sample_rate = torchaudio.load(self.path)
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        spectrogram = spectrogram.permute(0, 2, 1)
        spectrogram = spectrogram[0]
        transformer1 = transforms.ToPILImage()
        transformer2 = transforms.Resize((round(self.chart_generator.duration_time * 20), 128))
        transformer3 = transforms.ToTensor()
        spectrogram = transformer1(spectrogram)
        spectrogram = transformer2(spectrogram)
        spectrogram = transformer3(spectrogram)
        spectrogram = torch.squeeze(spectrogram, 0)
        return spectrogram


filename = "F:/Study/Project/Automatic-music-chart-generator/Resource/AITUS/music"

data_generator = DatasetGenerator(filename)
gram = data_generator.spectrogram
print("Shape of spectrogram:{}".format(gram.size()))
plt.figure()
plt.imshow(gram.numpy(), cmap='gray')
plt.show()
