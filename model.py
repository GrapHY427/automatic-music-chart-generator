import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 1
LR_G = 1e-5  # learning rate for generator
LR_D = 1e-5  # learning rate for discriminator
N_SPECTRUM = 5 * 1200 * 200  # think of this as number of ideas for generating an art work (Generator)
N_RESULT = (5 * 1200 + 1800 / 20) * 200
CHART_COMPONENTS = 1800 * 10  # it could be total point G can draw in the canvas


# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()


def human_charts():  # charts from the chart engineer (real target)
    pass


class Generation(nn.Module):
    def __init__(self):
        super(Generation, self).__init__()
        self.first_layer = nn.Sequential(  # Generator
            nn.Linear(N_SPECTRUM, 524288),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(524288, 262144),  # making a painting from these random ideas
        )
        self.drop_out_layer1 = nn.Dropout(0.2)
        self.second_layer = nn.Sequential(  # Generator
            nn.Linear(262144, 131072),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(131072, 65536),  # making a painting from these random ideas
        )
        self.drop_out_layer2 = nn.Dropout(0.2)
        self.third_layer = nn.Sequential(  # Generator
            nn.Linear(65536, 32768),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(32768, CHART_COMPONENTS),  # making a painting from these random ideas
        )

    def forward(self, input_spectrum):
        input_spectrum = self.first_layer(input_spectrum)
        input_spectrum = self.drop_out_layer1(input_spectrum)
        input_spectrum = self.second_layer(input_spectrum)
        input_spectrum = self.drop_out_layer2(input_spectrum)
        input_spectrum = self.third_layer(input_spectrum)
        return input_spectrum


class Discriminate(nn.Module):
    def __init__(self):
        super(Discriminate, self).__init__()
        self.first_layer = nn.Sequential(  # Generator
            nn.Linear(N_SPECTRUM, 524288),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(524288, 131072),  # making a painting from these random ideas
        )
        self.drop_out_layer1 = nn.Dropout(0.2)
        self.second_layer = nn.Sequential(  # Generator
            nn.Linear(131072, 32768),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(32768, 8192),  # making a painting from these random ideas
        )
        self.drop_out_layer2 = nn.Dropout(0.2)
        self.third_layer = nn.Sequential(  # Generator
            nn.Linear(8192, 2048),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(2048, 512),  # making a painting from these random ideas
        )
        self.drop_out_layer3 = nn.Dropout(0.2)
        self.forth_layer = nn.Sequential(  # Generator
            nn.Linear(512, 128),  # random ideas (could from normal distribution)
            nn.ReLU(),
            nn.Linear(128, 1),  # making a painting from these random ideas
            nn.Sigmoid(),  # tell the probability that the art work is made by artist
        )

    def forward(self, input_dataset):
        input_dataset = self.first_layer(input_dataset)
        input_dataset = self.drop_out_layer1(input_dataset)
        input_dataset = self.second_layer(input_dataset)
        input_dataset = self.drop_out_layer2(input_dataset)
        input_dataset = self.third_layer(input_dataset)
        input_dataset = self.drop_out_layer2(input_dataset)
        input_dataset = self.forth_layer(input_dataset)
        return input_dataset


G = Generation()

D = Discriminate()

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()  # something about continuous plotting

for step in range(10000):
    artist_paintings = human_charts()  # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_SPECTRUM, requires_grad=True)  # random ideas\n
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)
    prob_artist1 = D(G_paintings)  # D try to reduce this prob
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)  # D try to increase this prob
    prob_artist1 = D(G_paintings.detach())  # D try to reduce this prob
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    if step % 50 == 0:  # plotting
        pass
#         plt.cla()
#         plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
#         plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
#         plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
#         plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
#                  fontdict={'size': 13})
#         plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
#         plt.ylim((0, 3))
#         plt.legend(loc='upper right', fontsize=10)
#         plt.draw()
#         plt.pause(0.01)
#
# plt.ioff()
# plt.show()
