import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

N = 30
dataset = np.zeros((2*N, N+1)) # 0, 1, 2, .......... 29

for i in range(N):
    dataset[i][N]=0
    dataset[i][i]=1
        
for i in range(N, 2*N):
    dataset[i][N]=1
    dataset[i][i-N]=1
            
labels = np.zeros((2*N, N))
for i in range(N):
    one_idx = (i+10)%N
    labels[i][one_idx]=1

for i in range(N, 2*N):
    one_idx = (i-N+20)%N
    labels[i][one_idx]=1
            
# first half of the dataset is one relation-0 and the second half on relation-1
# last number in input represents the relation number which is then convert to a N-dim vector by the model, so that finally we have e_h and e_r of same dims

# shuffle the dataset
shuffle = np.random.permutation(2*N)
dataset = dataset[shuffle]
labels = labels[shuffle]

# divide in train and validation set
train_proportion = 0.8
train_data = dataset[:int(train_proportion*2*N)]
train_labels = labels[:int(train_proportion*2*N)]
val_data = dataset[int(train_proportion*2*N):]
val_labels = labels[int(train_proportion*2*N):]

# convert to tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).float()
val_data = torch.from_numpy(val_data).float()
val_labels = torch.from_numpy(val_labels).float()


class BilinearMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(2, input_size)
        self.bl = nn.Bilinear(input_size, input_size, hidden_size, bias=False)
        self.lin = nn.Linear(hidden_size, output_size, bias=False)
        
    def forward(self, x):
        e_r = self.embed(x[:, N].long())
        e_h = x[:, 0:N]
        h = self.bl(e_h, e_r)
        logits = self.lin(h)
        
        return logits

def train(model, train_data, train_labels, val_data, val_labels, epochs=100, batch_size=16, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []
    
    running_train_loss = 0
    print(epochs)
    for epoch in range(epochs):
        model.train()
        correct_train_preds = 0
        total_train_preds = 0
        for batch in range(0, len(train_data), batch_size):
            optimizer.zero_grad()
            output = model(train_data[batch:batch+batch_size])
            loss = loss_fn(output, torch.argmax(train_labels[batch:batch+batch_size], axis=1)) 
            running_train_loss += loss.item()
            preds = torch.argmax(output, axis=1)
            correct_train_preds += (preds == torch.argmax(train_labels[batch:batch+batch_size], axis=1)).sum().item()
            total_train_preds += len(preds)
            loss.backward()
            optimizer.step()
        model.eval()


        output = model(val_data)
        val_loss = loss_fn(output, torch.argmax(val_labels, axis=1)).item()
        val_preds = torch.argmax(output, axis=1)
        correct_val_preds = (val_preds == torch.argmax(val_labels, axis=1)).sum().item()
        total_val_preds = len(val_preds)
        avg_train_loss = running_train_loss / (len(train_data) / batch_size)
        train_acc = correct_train_preds / total_train_preds
        val_acc = correct_val_preds / total_val_preds
        train_loss_values.append(avg_train_loss)
        val_loss_values.append(val_loss)
        train_acc_values.append(train_acc)
        val_acc_values.append(val_acc)

        print("Epoch: {} | Train loss: {:.2f} | Validation loss: {:.2f} | Train accuracy: {:.2f} | Validation accuracy: {:.2f}".format(epoch, avg_train_loss, val_loss, train_acc, val_acc))

        running_train_loss = 0
    return model, train_loss_values, val_loss_values, train_acc_values, val_acc_values


model = BilinearMLP(N, 100, N)
model, train_loss_values, val_loss_values, train_acc_values, val_acc_values = train(model, train_data, train_labels, val_data, val_labels, epochs=10000, batch_size=16, lr=0.003)
