"""
chat.py - Interactive Text Generation & Chat with Kira
Compatible with checkpoints trained via train.py
"""

import os
import argparse
import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Chat or complete text with Kira LLM")
    parser.add_argument("--ckpt", type=str, default="out/ckpt.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/kira/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "complete"],
                        help="'chat' for conversational turn-taking, 'complete' for raw text completion")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Penalty for repeating tokens (1.0 = off)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cutoff (1.0 = off)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading checkpoint from {args.ckpt}...")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint {args.ckpt} not found! Please train the model first with train.py.")

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    bot_id = tokenizer.token_to_id("<|bot|>")

    print("\n" + "=" * 60)
    print("  Kira LLM - Interactive Session")
    print(f"  Mode: {args.mode.upper()} | Device: {device} | Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("  Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if args.mode == "chat":
            prompt_text = f"<|user|>\n{user_input}\n<|bot|>\n"
        else:
            prompt_text = user_input

        # Encode input prompt
        encoded = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=eot_id,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p,
                stop_token_ids=[i for i in (tokenizer.token_to_id("<|user|>"), bot_id) if i is not None],
            )

        # Slice off prompt tokens to only decode the response
        generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
        reply = tokenizer.decode(generated_tokens)

        # Clean up any trailing special tokens
        for special in ["<|endoftext|>", "<|user|>", "<|bot|>"]:
            reply = reply.replace(special, "")

        print(f"\nKira > {reply.strip()}\n")


if __name__ == "__main__":
    main()
