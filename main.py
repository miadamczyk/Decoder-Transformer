from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from model import DecoderOnlyTransformer, language_modeling_loss
from dataset import create_dataloaders

import wandb
import argparse
import os
import torch

# with grad clip as in original GPT paper
def train_one_epoch(model, dataloader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = language_modeling_loss(logits, y)

        loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, split_name="Val"):
    model.eval()
    total_loss = 0.0

    for x, y in tqdm(dataloader, desc=split_name, leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = language_modeling_loss(logits, y)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )

    train_loader, val_loader, test_loader, vocab_size, _, _ = create_dataloaders(
        file_path=args.data_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.block_size,
        dropout=args.dropout,
        attn_type=args.attn_type,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    if args.wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    save_dir = (
        f"{args.attn_type}_d{args.d_model}_h{args.n_heads}_l{args.n_layers}_bs{args.block_size}"
    )

    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best_model.pt")
    print(f"Saving checkpoints to: {best_model_path}")
    best_val_loss = 100000
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        scheduler.step()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
        )

        val_loss = evaluate(model, val_loader, device, split_name="Val")

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"✓ Saved new best model (val loss = {val_loss:.4f})")

    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(model, test_loader, device, split_name="Test")

    print("\n======================================")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("======================================")

    if wandb is not None:
        wandb.log({"test/loss": test_loss})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer on Shakespeare")

    parser.add_argument("--data_path", type=str, default="data/shakespeare.txt",
                        help="Path where Shakespeare dataset will be stored/downloaded")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Maximum sequence length (number of tokens) used for training samples")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of sequences processed in parallel during training")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction of the dataset used for training (remaining split into val/test)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of the dataset used for validation")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading (0 = main process)")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Transformer embedding dimension (model hidden size)")
    parser.add_argument("--n_heads", type=int, default=2,
                        help="Number of attention heads in multi-head self-attention")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of decoder blocks (Transformer layers)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability used throughout the model")
    parser.add_argument("--attn_type", type=str, choices=["custom", "native"], default="custom",
                        help="Type of attention implementation: 'custom' = manually implemented multi-head attention, "
                             "'native' = PyTorch built-in MultiheadAttention")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of full training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay (L2 regularization) coefficient")
    parser.add_argument('--step_size', default=5, type=int,
                        help='Scheduler step size')
    parser.add_argument('--gamma', default=0.1, type=int,
                        help='Scheduler gamma')
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping (0 == disabled)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (user or team name)")
    parser.add_argument("--wandb_project", type=str, default="decoder-only-transformer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional run name for Weights & Biases")


    args = parser.parse_args()
    main(args)
