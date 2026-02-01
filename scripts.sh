python main.py --attn_type custom --d_model 512 --n_heads 8 --n_layers 6 --block_size 128 --wandb

python generate.py --model_dir "models" --model_name "best_model.pt" --start_text "ROMEO: " --max_new_tokens 1000 --device cuda
