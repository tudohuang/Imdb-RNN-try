from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
review = input("Give me a sentence(less than 500)\nReview:")
path = input("Model paht:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

new_review = review
words = new_review.lower().split()
review_to_int = [word_index.get(word, word_index["<UNK>"]) for word in words]



maxlen = 500 
review_padded = pad_sequences([review_to_int], maxlen=maxlen)

review_tensor = torch.tensor(review_padded).long().to(torch.device('cuda'))
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

model = torch.load(path).to(device)

model.eval()  
with torch.no_grad(): 
    prediction = model(review_tensor)

predicted_class = 'Positive' if prediction.item() > 0.5 else 'Negative'
print(f'Predicted sentiment: {predicted_class}')
