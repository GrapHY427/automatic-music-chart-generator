from create_chart import *
import matplotlib.pyplot as plt


filename = "F:/Study/Project/Automatic-music-chart-generator/Resource/AITUS/music"

data_generator = DatasetGenerator(filename)
gram = data_generator.spectrogram
print("Shape of spectrogram:{}".format(gram.size()))
plt.figure()
plt.imshow(gram.numpy(), cmap='gray')
plt.show()

# chart = ChartParser('chart.hard')
# data = chart.get_note_list()
# for each in data:
#     print(each)


# util = ChartGenerator('music')
# util.generate_page_list()
# print(util.bpm)
# util.generate_chart()

# i = 0
# for each in util.page_list:
#     print(str(i) + str(each))
#     i += 1

# data = chart.create_note_list_tensor()
# print(data.size())
# for each1 in data:
#     for each2 in each1:
#         print(float(each2), end="    ")
#     print("")
