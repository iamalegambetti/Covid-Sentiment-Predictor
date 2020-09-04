import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab # To Load Pretrained Embeddings
import torch.optim as optim
import numpy as np


# Define Fields
TEXT = data.Field(tokenize = 'spacy', batch_first = True) # Text Tokenizer
LABEL = data.LabelField() # IMPORTANT! Do not change the datatype within this FIELD - Leave it as LONGTENSOR(defaault)
fields = {'cleaned_text': ('text', TEXT), 'label': ('label', LABEL)} # Field Path routines
path = '/Volumes/Transcend/CADD/covid-sentiment/training-data/upgraded/' # Absolute Path of the data



# Load Train Test Data w/ Iterator
train_data, test_data = data.TabularDataset.splits(
                            path = path,
                            train = 'train.csv',
                            test = 'test.csv',
                            format = 'csv',
                            fields = fields)


# Load Custom Embeddings
custom_embeddings = vocab.Vectors(name = '/Volumes/Transcend/CADD/embeddings/word2vec_twitter_128.txt',
                                  cache = 'custom_embeddings',
                                  unk_init = torch.Tensor.normal_)


# Build Vocabulary
MAX_VOCAB_SIZE = 30_000 # Use N most frequent words
TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = custom_embeddings)
LABEL.build_vocab(train_data)


# Build Iterators
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

sort_key = lambda x: len(x.cleaned_text), sort_within_batch = False


# Model
class CNN_bi_uni_gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, output_classes):
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        # Bi-Gram Convolutional Layer
        self.conv_uni = nn.Conv2d(1, embedding_dim, (1, embedding_dim))
        self.conv_big = nn.Conv2d(1, embedding_dim, (2, embedding_dim))

        # Linear Layer
        self.fc = nn.Linear(embedding_dim, output_classes)

        # Droupout Layer
        self.droupout = nn.Dropout(p=0.3)

    def forward(self, text):

        # Store in embedding
        emb = self.embedding(text).unsqueeze(1) # Add one extra dimension to fit the convolution

        # Convolve and Activate
        bigram = F.relu(self.conv_big(emb).squeeze(3))
        unigram = F.relu(self.conv_uni(emb).squeeze(3))

        # Pool - Maxpool
        pooled_bi = F.max_pool1d(bigram, bigram.shape[2]).squeeze(2)
        pooled_uni = F.max_pool1d(unigram, unigram.shape[2]).squeeze(2)

        # Concat
        concat = self.droupout(torch.cat((pooled_uni), dim = 1))

        return concat



# TRAIN THE MODEL

# Init the model
vocabular_dim = MAX_VOCAB_SIZE + 2
model = CNN_bi_uni_gram(vocabular_dim, 128, 1, 3)

# Fit Pretrained embeddings in the model
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Init to 0 the initial weights of unkwnown and padding tokens
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(128)
model.embedding.weight.data[1] = torch.zeros(128)

# Criterion and Optim
optimizer = optim.Adam(model.parameters(), lr = 1e-3) # Optimizer - lr = learning rate
criterion = nn.CrossEntropyLoss() # Loss
model = model.to(device) # Model to device - cpu or gpu (if available)
criterion = criterion.to(device) # Loss to device

# Train here
model.train()
EPOCHS = 2

for epoch in range(EPOCHS):

    losses = []
    for i, batch in enumerate(train_iterator):
        optimizer.zero_grad() # Clean Gradients
        preds = model(batch.text)

        loss = criterion(preds, batch.label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'Current average loss at epoch {epoch + 1} is {np.mean(losses)}.')


# Test Set Iterator
test_iter = data.Iterator(test_data, batch_size=BATCH_SIZE, device=-1, sort=False, sort_within_batch=False, repeat=False)

# Method for Accuracy
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch.
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y) # Check if predicted are equal to the correct label
    return correct.sum() / torch.FloatTensor([y.shape[0]])


# TEST SET EVALUATION
epoch_acc = 0
with torch.no_grad():
    model.eval()
    for batch in test_iter:
        predictions = model(batch.text)
        acc = categorical_accuracy(predictions, batch.label)
        epoch_acc += acc.item()

print('Test Set Accuracy: ',epoch_acc / len(test_iter))


# TRAIN SET EVALUATION
epoch_acc = 0
with torch.no_grad():
    model.eval()
    for batch in train_iterator:
        predictions = model(batch.text)
        acc = categorical_accuracy(predictions, batch.label)
        epoch_acc += acc.item()

print('Train Set Accuracy: ',epoch_acc / len(train_iterator))

# Save Model
torch.save(model.state_dict(), '/Volumes/Transcend/CADD/covid-sentiment/predictions/cnn-uni-classifier.pt')
