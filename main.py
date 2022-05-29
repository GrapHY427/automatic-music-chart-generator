import torchaudio
from matplotlib import pyplot as plt


filename = "F:/Study/Project/Automatic-music-chart-generator/Resource/AITUS/music"

waveform, sample_rate = torchaudio.load(filename)
print("Shape of waveform:{}".format(waveform.size()))  # 音频大小
print("sample rate of waveform:{}".format(sample_rate))  # 采样率

spectrogram = torchaudio.transforms.Spectrogram()(waveform)
spectrogram = spectrogram.permute(2, 1, 0)
print("Shape of spectrogram:{}".format(spectrogram.size()))
plt.figure()
plt.imshow(spectrogram.log2()[:, :, 0].numpy(), cmap='gray')
plt.show()