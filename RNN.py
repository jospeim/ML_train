# ===============================
# 0️⃣ Importations
# ===============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Exemple de DataFrame d'entraînement et de test
# ===============================
name_test = "C:/Users/joach/OneDrive/Documents/entrainement/test_hmp/TF1_test_set.xlsx"
name = "C:/Users/joach/OneDrive/Documents/entrainement/test_hmp/TF1_train_set.csv"

df_train = pd.read_csv(name, sep = ";").iloc[:500, :]
df_test = pd.read_excel(name_test).iloc[:500, :]
df_test = df_test.dropna()

cols = df_train.columns.drop("visitor_id")

df_train[cols] = df_train[cols].apply(lambda x: x % 7)
# ===============================
# 2️⃣ Création des séquences glissantes
# ===============================
def create_sequences_from_df(df, seq_length=10):
    X, y = [], []
    for _, row in df.iterrows():
        values = row.dropna().values[1:]  # on ignore la colonne 'id'
        if len(values) <= seq_length:
            continue
        for i in range(len(values) - seq_length):
            seq = values[i:i + seq_length]
            target = int(values[i + seq_length])
            X.append(seq)
            y.append(target)
    return np.array(X, dtype=float), np.array(y, dtype=int)

X_train, y_train = create_sequences_from_df(df_train, seq_length=3)
X_test, y_test = create_sequences_from_df(df_test, seq_length=3)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ===============================
# 3️⃣ Dataset et DataLoader
# ===============================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch, seq_len, 1)
        self.y = torch.tensor(y, dtype=torch.long)                   # entiers pour CrossEntropy
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=8, shuffle=False)

# ===============================
# 4️⃣ Modèle RNN de classification
# ===============================
class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, n_classes=7):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]          # dernière sortie temporelle
        logits = self.fc(out)
        return logits

model = RNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===============================
# 5️⃣ Boucle d'entraînement avec suivi accuracy
# ===============================
n_epochs = 150
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []

for epoch in range(n_epochs):
    # ---- Entraînement ----
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    train_acc = correct / total
    train_loss = total_loss / total

    # ---- Test ----
    # ---- Test ----
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    if total > 0:
        test_loss = total_loss / total
        test_acc = correct / total
    else:
        test_loss, test_acc = np.nan, np.nan
        print(f"[!] Aucune séquence valide dans df_test (total={total})")
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | TrainAcc={train_acc*100:.1f}% | TestAcc={test_acc*100:.1f}% | TrainLoss={train_loss:.4f} | TestLoss={test_loss:.4f}")

# ===============================
# 6️⃣ Visualisation de la convergence
# ===============================
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(test_loss_list, label="Test Loss")
plt.legend(); plt.title("Évolution de la loss")

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.legend(); plt.title("Évolution de l'accuracy")
plt.show()

# ===============================
# 7️⃣ Prédiction finale sur df_test
# ===============================
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).numpy()

df_pred = pd.DataFrame({
    "y_true": y_test,
    "y_pred": preds
})
df_pred["correct"] = (df_pred["y_true"] == df_pred["y_pred"]).astype(int)

print("\nPrédictions sur df_test :")
print(df_pred)
print(f"\nAccuracy finale test : {df_pred['correct'].mean()*100:.2f}%")
