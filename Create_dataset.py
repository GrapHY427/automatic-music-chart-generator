from create_chart import *


filename = "F:/Study/Project/Automatic-music-chart-generator/Resource/AITUS/music"

data_generator = DatasetGenerator(filename)
gram = data_generator.spectrogram
print("Shape of spectrogram:{}".format(gram.size()))
plt.figure()
plt.imshow(gram.numpy(), cmap='gray')
plt.show()
