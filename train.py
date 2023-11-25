import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
path = input("Path_for_model:")
# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Preprocess the data: padding sequences
maxlen = 500  # Cut texts after this number of words
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Convert the data to PyTorch tensors
x_train_torch = torch.tensor(x_train).long()
y_train_torch = torch.tensor(y_train).float()  # BCELoss 需要 float 張量
x_test_torch = torch.tensor(x_test).long()
y_test_torch = torch.tensor(y_test).float()

# Release the original data to save memory
del x_train, y_train, x_test, y_test

# Define the RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(max_features, 32)
        self.rnn = nn.RNN(32, 64, batch_first=True)
        self.dense = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.dense(x[:, -1, :])
        return self.sigmoid(x)

model = RNN().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Prepare DataLoader for batch processing
batch_size = 32
train_data = TensorDataset(x_train_torch, y_train_torch)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100  # Reduced number of epochs
for epoch in range(num_epochs):
    for inputs, labels in tqdm(train_loader):
        # Forward pass
        outputs = model(inputs.to(device))
        loss = criterion(outputs.squeeze(), labels.to(device))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model, path)
