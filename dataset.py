import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

def download_shakespeare(file_path):
    if os.path.exists(file_path):
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print("Shakespeare dataset not found.")
    print("Downloading Tiny Shakespeare dataset...")

    urllib.request.urlretrieve(SHAKESPEARE_URL, file_path)

    print(f"Downloaded dataset to {file_path}")

class ShakespeareDataset(Dataset):
    """
    Produces (input, target) pairs:
    input  = text[i : i + block_size]
    target = text[i + 1 : i + block_size + 1]
    """

    def __init__(self, text, stoi, itos, block_size):
        super().__init__()

        # how many tokens model looks at
        self.block_size = block_size
        # dictionaries of string: int / int: string of used vocabulary
        self.stoi = stoi
        self.itos = itos

        self.data = torch.tensor(
            [stoi[c] for c in text],
            dtype=torch.long,
        )

    def __len__(self):
        # - is a buffer to make taking self.block_size + 1 possible (look at __getitem__ implementation)
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def decode(self, token_ids):
        return "".join(self.itos[int(i)] for i in token_ids)

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)


def create_datasets(
    file_path,
    block_size,
    train_ratio=0.9,
    val_ratio=0.05,
):
    download_shakespeare(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build vocabulary from full corpus
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    n = len(text)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    train_ds = ShakespeareDataset(train_text, stoi, itos, block_size)
    val_ds = ShakespeareDataset(val_text, stoi, itos, block_size)
    test_ds = ShakespeareDataset(test_text, stoi, itos, block_size)

    return train_ds, val_ds, test_ds, vocab_size, stoi, itos


def create_dataloaders(
    file_path,
    block_size,
    batch_size,
    train_ratio=0.9,
    val_ratio=0.05,
    num_workers=0,
):
    train_ds, val_ds, test_ds, vocab_size, stoi, itos = create_datasets(
        file_path=file_path,
        block_size=block_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_size, stoi, itos
