import torch
import torch.nn as nn
from torch.autograd import Variable


class MLPEmbedding(nn.Module):
  def __init__(self, n_users, n_beers, device, hidden_size=100):
    super().__init__()
    self.users_emb = nn.Embedding(n_users, hidden_size).to(device)
    self.beers_emb = nn.Embedding(n_beers, hidden_size).to(device)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_size*2, 200),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(200, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(50, 25),
        nn.BatchNorm1d(25),
        nn.ReLU(),
        nn.Dropout(0.6)
    ).to(device)
       
    self.last_layer = nn.Linear(25,1)
    self.device = device

  def forward(self, df):
    user_idx = torch.LongTensor(df.user_id.to_numpy()).to(self.device)
    beer_idx = torch.LongTensor(df.beer_id.to_numpy()).to(self.device)
    input = torch.cat((self.users_emb(user_idx), self.beers_emb(beer_idx)), 1).to(self.device)
    out = self.mlp(input)
    return out

  def predict(self, df):
    return torch.sigmoid(self.last_layer(self.forward(df)))
    
    
  def loss(self, train_data, loss_fn):
    y_pred = self.last_layer(self.forward(train_data)).view(-1)
    y_train = torch.Tensor(train_data.relevant.to_numpy()).to(self.device)
    
    return loss_fn(y_pred, y_train)
    
class GMF(nn.Module):
  def __init__(self, n_users, n_items, n_factors=5):
    self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    super().__init__()
    self.user_emb = nn.Embedding(n_users,n_factors).to(self.device)
    self.item_emb = nn.Embedding(n_items,n_factors).to(self.device)
    self.h_out = nn.Linear(n_factors,1)
  
  def forward(self, user, item):
    p = self.user_emb(user).to(self.device)
    q = self.item_emb(item).to(self.device)
    return torch.flatten(self.h_out(p*q))

  def predict(self, user, item):
    return torch.sigmoid(self.forward(user, item))

  def forward_no_h(self, df):
    user_id = Variable(torch.LongTensor(df.user_id.to_numpy())).to(self.device)
    beer_id = Variable(torch.LongTensor(df.beer_id.to_numpy())).to(self.device)
    p = self.user_emb(user_id)
    q = self.item_emb(beer_id)
    return p*q 