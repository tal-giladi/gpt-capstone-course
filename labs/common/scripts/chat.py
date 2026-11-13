"""Generate text from an HF-format model (base, SFT or DPO).

  python scripts/chat.py --model hf/my-gpt-350m --prompt "The history of the printing press"   # base: plain completion
  python scripts/chat.py --model hf/my-gpt-350m-sft --chat --prompt "Explain photosynthesis simply."
  python scripts/chat.py --model hf/my-gpt-350m-dpo --chat           # interactive, Ctrl-D / 'exit' to quit
"""
from __future__ import annotations

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default=None, help="one-shot; omit for an interactive loop")
    p.add_argument("--chat", action="store_true", help="wrap prompts with the chat template (SFT/DPO models)")
    p.add_argument("--system", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def load(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    return tok, model, device


def generate(tok, model, device, messages_or_text, args) -> str:
    import torch
    if isinstance(messages_or_text, list):
        ids = tok.apply_chat_template(messages_or_text, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"]
    else:
        ids = tok(messages_or_text, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                             temperature=args.temperature or None, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
    tok, model, device = load(args.model)
    history = [{"role": "system", "content": args.system}] if args.system else []

    def respond(text: str) -> str:
        if not args.chat:
            return generate(tok, model, device, text, args)
        history.append({"role": "user", "content": text})
        reply = generate(tok, model, device, list(history), args)
        history.append({"role": "assistant", "content": reply})
        return reply

    if args.prompt is not None:
        print(respond(args.prompt))
        return
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if text in ("exit", "quit"):
            break
        if text:
            print("model>", respond(text))


if __name__ == "__main__":
    main()
