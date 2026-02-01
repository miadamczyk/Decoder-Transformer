#!/usr/bin/env python3
import os
import torch
import argparse
from model import DecoderOnlyTransformer
from dataset import create_datasets

def generate_text(model, start_text, stoi, itos, device, block_size, max_new_tokens=500):
    model.eval()
    with torch.no_grad():
        generated = torch.tensor([stoi[c] for c in start_text], dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            input_ids = generated[:, -block_size:]
            logits = model(input_ids)

            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        output_text = "".join([itos[int(i)] for i in generated[0]])

    return output_text


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(args.model_dir, args.model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_args = checkpoint.get('args', None)

    d_model = model_args.get('d_model', 128)
    n_layers = model_args.get('n_layers', 2)
    n_heads = model_args.get('n_heads', 2)
    block_size = model_args.get('block_size', 128)
    dropout = model_args.get('dropout', 0.2)
    attn_type = model_args.get('attn_type', 'custom')

    _, _, _, vocab_size, stoi, itos = create_datasets(
        file_path=args.data_path,
        block_size=block_size
    )

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=block_size,
        dropout=dropout,
        attn_type=attn_type
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    text = generate_text(model, args.start_text, stoi, itos, device, block_size)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"output_{args.model_name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Generated text saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Shakespeare-like text from a trained Transformer.")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory where the checkpoint is saved")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved .pt model")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory where the checkpoint is saved")
    parser.add_argument("--data_path", type=str, default="data/shakespeare.txt", help="Path to training data (only if vocab reconstruction needed)")
    parser.add_argument("--start_text", type=str, default="ROMEO: ", help="Seed text to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on (cuda or cpu)")

    args = parser.parse_args()
    main(args)
