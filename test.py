from torch import nn, tensor, optim
import torch

# hyperparam
learning_rate=0.01

#linear_layer = nn.Linear(in_features=3, out_features=3)

# activation
relu = nn.ReLU()
gelu = nn.GELU()

# dropout 50% of parameters
dropout_layer = nn.Dropout(p=0.5)

linear_layer_2 = nn.Linear(in_features=3, out_features=1)

#y_hat = linear_layer_2(relu(linear_layer(X)))

class LinRegModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear_layer(x)
    
model = LinRegModel(in_features=1, out_features=1)
print(model)


# optimizing
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.MSELoss()

# training loop
epochs = 100

X = tensor([1.])

y_true = tensor([5.])

for n in range(epochs):
    # perform forwar pass - calls forward()
    y_hat = model(X)
    loss = loss_fn(y_hat, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if n % 10 == 0:
        print(f'Epoch {n:02d}: Loss={loss.item():.4f}')


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ffn_dim):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, ffn_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(ffn_dim, embedding_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)

"""
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
"""