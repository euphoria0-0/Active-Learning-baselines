import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes, in_features=2048, MLPlayersizes=[300,300,300], model='others'):
        super(MLP, self).__init__()
        self.MLPlayer1 = nn.Linear(in_features, MLPlayersizes[0])
        self.MLPlayer2 = nn.Linear(MLPlayersizes[0], MLPlayersizes[1])
        self.MLPlayer3 = nn.Linear(MLPlayersizes[1], MLPlayersizes[2])
        self.classifier = nn.Linear(MLPlayersizes[2], num_classes)
        self.embSize = MLPlayersizes[0]
        self.model = model.lower()

    def forward(self, x):
        x1 = nn.Tanh()(self.MLPlayer1(x))
        x2 = nn.Tanh()(self.MLPlayer2(x1))
        x3 = nn.Tanh()(self.MLPlayer3(x2))
        x = self.classifier(x3)
        if self.model == 'll':
            return x, [x1, x2, x3]
        elif self.model in ['badge', 'coreset', 'maxentropy', 'random']:
            return x, x3
        else:
            return x

    def get_embedding_dim(self):
        return self.embSize
