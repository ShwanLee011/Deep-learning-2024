import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchsummary import summary
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加載數據
with open('shakespeare_train.txt', 'r', encoding='utf8') as f:
    train_text = f.read()
with open('shakespeare_valid.txt', 'r', encoding='utf8') as f:
    valid_text = f.read()

# 建立詞彙表與映射
vocab = sorted(set(train_text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# 數據轉換為整數表示
train_data = np.array([vocab_to_int[c] for c in train_text], dtype=np.int32)
valid_data = np.array([vocab_to_int[c] for c in valid_text], dtype=np.int32)

# 超參數
embedding_dim = 64
hidden_dim_list = [64, 128, 256]
sequence_lengths = [50, 100, 200]
batch_size = 64
num_epochs = 20
learning_rate = 0.001

# 自定義數據集類
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 構建模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

# 訓練過程
def train_and_evaluate(hidden_dim, seq_length):
    print(f"\nInitializing model with hidden_dim={hidden_dim}, seq_length={seq_length}")
    train_dataset = TextDataset(train_data, seq_length)
    valid_dataset = TextDataset(valid_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    vocab_size = len(vocab)
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
    
    # 模型摘要
    print("\nModel Summary:")
    summary(model, input_size=(seq_length,), device="cuda" if device == 0 else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch, y_batch
            optimizer.zero_grad()
            
            # 初始化 hidden 狀態
            hidden = model.init_hidden(x_batch.size(0))
            
            output, hidden = model(x_batch, hidden)
            hidden = tuple(h.detach() for h in hidden)  # 避免梯度爆炸
            loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # 驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch, y_batch
                hidden = model.init_hidden(x_batch.size(0))
                output, hidden = model(x_batch, hidden)
                loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
                val_loss += loss.item()
        val_loss /= len(valid_loader)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, history

# 主函數
if __name__ == "__main__":
    hidden_dim = 128  # 選擇隱藏層維度
    seq_length = 100  # 選擇序列長度
    # 初始化數據集
    train_dataset = TextDataset(train_data, seq_length)
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
    # 設定起始詞和生成參數
    start_text = "JULIET"
    gen_length = 1000  # 生成文本的總字符數
    print(f"Generating text starting with: {start_text}\n")
    
    # 調用生成函數
    generated_text = model.generate_text(
        start_text=start_text, 
        length=gen_length, 
        vocab_to_int=vocab_to_int, 
        int_to_vocab=int_to_vocab, 
        device=device
    )
    
    # 分行顯示生成的文本
    print(f"Generated text:\n{generated_text}")