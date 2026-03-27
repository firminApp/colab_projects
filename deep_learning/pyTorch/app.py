import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader    
import torch.nn as nn
#  preparer le dataset
# utiliser train test split de skilearn pour separer les données en deux parties, une pour l'entrainement et une pour le test
#  a creer:
#  train_dataset, test_dataset , train_loader, test_loader
class dataset(Dataset):
    def __init__(self,csv_path)->None:
       super().__init__()
       self.data = pd.read_csv(csv_path)
       self.data.fillna(self.data.mean(), inplace=True)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # reccuperer les featuer de 0 a n-1 et la target n
        features=torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.int32)
        target=torch.tensor(self.data.iloc[idx, -1], dtype=torch.int32)
        return features, target
    



#  definir le model
class nn_model(nn.Module):
    def __init__(self)->None:
        super().__init__()
        # activation function
        self.relu = nn.ReLU()
        # definir les couches lineaires
        self.linear1 = nn.Linear(in_features=9, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=8)   
        self.linear3 = nn.Linear(in_features=8, out_features=1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x
#  COURS ENDED HERE FOR TODAY: 27/06/2024






#  definir la fonction d'entrainement
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    #  activer le mode entrainement du model
    # iterer sur les batches de données
    # pour chaque batch, faire une passe avant,
    #  calculer la perte, 
    # reinitialiser les gradients,
    # calculer les gradients en faisant une passe arrière,
    #  mettre à jour les poids du modèle
    total_loss = 0
    for features, target in train_loader:
        optimizer.zero_grad()
        outputs = model(features.float())
        loss = criterion(outputs, target.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Average Loss: {avg_loss:.4f}')
#  definir la fonction d'evaluation
#  utiliser torch metrics pour calculer les métriques d'évaluation telles que l'accuracy, la precision, le recall et le f1-score
#  suivre le loss sur l'evaluation
# tracer la courbe du loss par epoch pour visualiser l'entrainement du model
class eval:
    def __init__(self, model, test_loader, criterion):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
    
    def evaluate(self):
        with torch.no_grad():
            total_loss = 0
            for features, target in self.test_loader:
                outputs = self.model(features.float())
                loss = self.criterion(outputs, target.float().unsqueeze(1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        print(f'Average Loss: {avg_loss:.4f}')
#  definiir la fonction main

if __name__ == "__main__":
    my_dataset = dataset("./water_potability.csv")
    train_loader = DataLoader(my_dataset, batch_size=8, shuffle=True)
    features, target = next(iter(train_loader))
    print(features, target)
    model=nn_model()
    print(model)
    # print(len(my_dataset))