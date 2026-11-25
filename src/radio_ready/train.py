"""Train and run the 1.5B rewriter on the SFT examples of radio_ready.sft (LoRA, own loop).

Written for this project: the LoRA layers, the trainable control-token embeddings, the
completion-only loss and the training loop. Only the base model comes from Hugging Face
(default Qwen/Qwen2.5-1.5B-Instruct).

Data: data/processed/sft/<name>.jsonl from radio_ready.sft (id, source, split, controls,
prompt, completion) and its <name>.tokens.json. Each sequence is prompt + completion + EOS,
and the loss is the negative log-likelihood of the completion tokens only:
    L = sum over completion positions j of -log P(s_j | s_<j)
Training uses split == train (val for the validation loss); test-split examples and the
Genius sections reserved for commercial-LLM evaluation are never trained on. --sources picks
the pair sources (paradehate for stage 1; genius kidzbop for stage 2).

Control tokens (<edit>, <syl:N>, <rhyme:X>) are added to the tokenizer. Their embeddings are
new trainable vectors (initialized to the mean input embedding), so the model can learn what
they mean; LoRA alone cannot, because the base embedding matrix stays frozen.

LoRA (Hu et al., 2022): every attention and MLP projection W gets a trainable update
(alpha / r) * B A, A in R^{r x d_in}, B in R^{d_out x r}, B = 0 at the start (so training
starts from the pretrained model); W stays frozen.

Device: CUDA (bfloat16), then Apple MPS (float32), then CPU (float32).

Commands:
  train    python -m radio_ready.train train --data sft_edit_syl_rhyme --sources paradehate \
               --name stage1
           python -m radio_ready.train train --data sft_edit_syl_rhyme --sources genius kidzbop \
               --resume stage1 --name stage2 --epochs 3
  predict  python -m radio_ready.train predict --adapter stage2 [--input FILE] [--samples 8]

Outputs:
  outputs/sft_models/<name>/  lora.pt (LoRA and control-token weights of the best validation
                              loss), config.json, train_log.jsonl (the loss curve)
  outputs/systems/sft-<name>_<input stem>.jsonl  rewrites of the evaluation sections, in the
                              radio_ready.generate format (id, lines, valid), for
                              radio_ready.compare. Not in outputs/generations, so the vetting
                              app does not list them; copy one there to vet it by hand.

Smoke test: add --limit 200 --max-steps 20 to train and --limit 5 to predict.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from radio_ready import paths
from radio_ready.eval_set import EVAL_SECTIONS_JSONL, SYSTEMS_DIR

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS_DIR = paths.OUTPUTS / "sft_models"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IGNORE = -100


# --- model pieces -------------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B (A dropout(x)); W frozen, A and B trained."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        update = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + (self.scale * update).to(x.dtype)


class ControlEmbedding(nn.Module):
    """The frozen input embedding, except for the control tokens, which get trainable vectors."""

    def __init__(self, base: nn.Embedding, new_ids: list[int]):
        super().__init__()
        self.base = base
        mean = base.weight.detach().float().mean(0)
        self.ctrl_weight = nn.Parameter(mean.repeat(len(new_ids), 1).to(base.weight.device))
        lookup = torch.full((base.num_embeddings,), -1, dtype=torch.long)
        lookup[torch.tensor(new_ids)] = torch.arange(len(new_ids))
        self.register_buffer("lookup", lookup.to(base.weight.device), persistent=False)

    @property
    def weight(self):  # for code that reads the embedding matrix (e.g. tied heads)
        return self.base.weight

    def forward(self, ids):
        out = self.base(ids)
        slot = self.lookup[ids]
        mask = slot >= 0
        if mask.any():
            out = out.clone()
            out[mask] = self.ctrl_weight[slot[mask]].to(out.dtype)
        return out


def add_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> int:
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in LORA_TARGETS and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
                replaced += 1
    return replaced


def trained_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ".lora_" in k or "ctrl_weight" in k}


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(cfg: dict, adapter: Path | None, train: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device()
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_tokens(cfg["tokens"], special_tokens=True)  # works across transformers versions
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=dtype).to(device)
    if len(tok) > model.get_input_embeddings().num_embeddings:  # Qwen has spare rows: rarely
        model.resize_token_embeddings(len(tok))
    for p in model.parameters():
        p.requires_grad_(False)
    new_ids = tok.convert_tokens_to_ids(cfg["tokens"])
    if new_ids:
        model.set_input_embeddings(ControlEmbedding(model.get_input_embeddings(), new_ids))
    n = add_lora(model, cfg["r"], cfg["alpha"], cfg["dropout"] if train else 0.0)
    if adapter is not None:
        state = torch.load(adapter / "lora.pt", map_location="cpu")
        missing = set(trained_state(model)) - set(state)
        if missing:
            raise ValueError(f"{adapter}: missing {len(missing)} trained tensors")
        model.load_state_dict(state, strict=False)
    if train:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{device}, {dtype}: LoRA on {n} layers, {len(new_ids)} control tokens; "
          f"{trainable:,} trainable parameters", flush=True)
    return tok, model, device


# --- data ---------------------------------------------------------------------------------------

def load_examples(name: str, sources: list[str] | None, limit: int | None):
    from radio_ready.commercial import reserved_ids

    path = paths.SFT_DIR / f"{name}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    tokens = json.loads((paths.SFT_DIR / f"{name}.tokens.json").read_text(encoding="utf-8"))
    reserved = reserved_ids()
    keep = [r for r in rows if (not sources or r["source"] in sources) and r["id"] not in reserved]
    split = {s: [r for r in keep if r["split"] == s] for s in ("train", "val")}
    random.Random(0).shuffle(split["train"])
    if limit:
        split["train"] = split["train"][:limit]
    controls = next((r["controls"] for r in keep), [])
    print(f"{name}: train {len(split['train'])}, val {len(split['val'])} "
          f"(sources {sources or 'all'}; {len(rows) - len(keep)} other-source or reserved left out)")
    return split["train"], split["val"], tokens, controls


def encode(tok, ex: dict, max_len: int) -> dict | None:
    """Token ids with labels = IGNORE on the prompt (loss on the completion only)."""
    src = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    tgt = tok(ex["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(src) + len(tgt) > max_len:
        return None
    return {"input_ids": src + tgt, "labels": [IGNORE] * len(src) + tgt}


def collate(batch: list[dict], pad_id: int, device) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), IGNORE)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], labels[i, :n], mask[i, :n] = (torch.tensor(b["input_ids"]),
                                                  torch.tensor(b["labels"]), 1)
    return {"input_ids": ids.to(device), "labels": labels.to(device),
            "attention_mask": mask.to(device)}


def completion_nll(model, batch: dict) -> tuple[torch.Tensor, int]:
    """Summed NLL of the completion tokens and their count."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits, labels = logits[:, :-1].float(), batch["labels"][:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                          ignore_index=IGNORE, reduction="sum")
    return nll, int((labels != IGNORE).sum())


@torch.no_grad()
def evaluate(model, data: list[dict], pad_id: int, device, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        nll, n = completion_nll(model, collate(data[i : i + batch_size], pad_id, device))
        total, count = total + nll.item(), count + n
    model.train()
    return total / max(count, 1)


# --- training -----------------------------------------------------------------------------------

def train(args) -> None:
    out = MODELS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    train_ex, val_ex, tokens, controls = load_examples(args.data, args.sources, args.limit)
    resume = MODELS_DIR / args.resume if args.resume else None
    cfg = {"base": args.base, "r": args.r, "alpha": args.alpha, "dropout": args.dropout}
    if resume:
        prev = json.loads((resume / "config.json").read_text())
        cfg.update({k: prev[k] for k in ("base", "r", "alpha")})
        tokens = sorted(set(tokens) | set(prev["tokens"]))
    cfg.update({"tokens": sorted(tokens), "controls": controls, "data": args.data,
                "sources": args.sources, "resume": args.resume})
    if resume and set(cfg["tokens"]) != set(prev["tokens"]):
        # control tokens new to this stage: load the old ones by name into the new layout
        old = torch.load(resume / "lora.pt", map_location="cpu")
        ctrl = [k for k in old if "ctrl_weight" in k]
        tok, model, device = load_model(cfg, None, train=True)
        model.load_state_dict({k: v for k, v in old.items() if k not in ctrl}, strict=False)
        emb = model.get_input_embeddings()
        with torch.no_grad():
            for i, t in enumerate(prev["tokens"]):
                emb.ctrl_weight[cfg["tokens"].index(t)] = old[ctrl[0]][i]
    else:
        tok, model, device = load_model(cfg, resume, train=True)
    (out / "lora.pt").unlink(missing_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    enc = [e for e in (encode(tok, x, args.max_len) for x in train_ex) if e]
    val = [e for e in (encode(tok, x, args.max_len) for x in val_ex) if e][: args.max_val]
    print(f"encoded: train {len(enc)} ({len(train_ex) - len(enc)} longer than {args.max_len} "
          f"tokens left out), val {len(val)}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = min(args.max_steps or 10**9,
                args.epochs * math.ceil(len(enc) / (args.batch_size * args.grad_accum)))
    warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))

    log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    rng = random.Random(args.seed)
    best = evaluate(model, val, tok.pad_token_id, device, args.batch_size) if val else math.inf
    log.write(json.dumps({"step": 0, "val_loss": round(best, 4)}) + "\n")
    print(f"step 0: val loss {best:.4f}", flush=True)
    step, started = 0, time.time()
    while step < total:
        order = list(range(len(enc)))
        rng.shuffle(order)
        chunk = args.batch_size * 50  # sort by length inside chunks: little padding
        batches = [sorted(order[c : c + chunk], key=lambda i: len(enc[i]["input_ids"]))[j : j + args.batch_size]
                   for c in range(0, len(order), chunk) for j in range(0, min(chunk, len(order) - c), args.batch_size)]
        rng.shuffle(batches)
        for k in range(0, len(batches), args.grad_accum):
            group = batches[k : k + args.grad_accum]
            tokens_in_step = sum(sum(l != IGNORE for l in enc[i]["labels"]) for b in group for i in b)
            loss_sum = 0.0
            for b in group:
                nll, _ = completion_nll(model, collate([enc[i] for i in b], tok.pad_token_id, device))
                (nll / tokens_in_step).backward()  # mean over the step's completion tokens
                loss_sum += nll.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            entry = {"step": step, "train_loss": round(loss_sum / tokens_in_step, 4),
                     "lr": sched.get_last_lr()[0]}
            if val and (step % args.eval_every == 0 or step == total):
                entry["val_loss"] = round(evaluate(model, val, tok.pad_token_id, device,
                                                   args.batch_size), 4)
                if entry["val_loss"] < best:
                    best = entry["val_loss"]
                    torch.save(trained_state(model), out / "lora.pt")
                print(f"step {step}/{total}: train {entry['train_loss']:.4f} "
                      f"val {entry['val_loss']:.4f} ({(time.time() - started) / 60:.1f} min)",
                      flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step >= total:
                break
    if not (out / "lora.pt").exists():  # no validation data, or it never improved
        torch.save(trained_state(model), out / "lora.pt")
    cfg.update({"steps": step, "lr": args.lr, "batch_size": args.batch_size * args.grad_accum,
                "max_len": args.max_len, "best_val_loss": round(best, 4),
                "minutes": round((time.time() - started) / 60, 1)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"best val loss {best:.4f}; adapter: {out}")


# --- prediction ---------------------------------------------------------------------------------

def parse_completion(text: str, n: int) -> list[str] | None:
    lines = [l.strip() for l in text.strip().splitlines()]
    return lines if len(lines) == n and all(lines) else None


@torch.no_grad()
def predict(args) -> None:
    from radio_ready.explicit import ExplicitDetector
    from radio_ready.sft import example

    adapter = MODELS_DIR / args.adapter
    cfg = json.loads((adapter / "config.json").read_text())
    tok, model, device = load_model(cfg, adapter, train=False)
    tok.padding_side = "left"
    detector = ExplicitDetector()
    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    records = records[: args.limit] if args.limit else records
    selector = None
    if args.rerank:
        from radio_ready.generate import Selector

        selector = Selector(detector)
    samples = max(1, args.samples)
    out = SYSTEMS_DIR / f"sft-{args.adapter}{'-rerank' if args.rerank else ''}" \
                        f"_{Path(args.input).stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(0, len(records), args.batch_size):
            batch = records[i : i + args.batch_size]
            # the prompt exactly as radio_ready.sft writes it, controls from the original lines
            prompts = [example({"id": r["id"], "source": "genius", "split": "test",
                                "original": r["lines"], "rewrite": r["lines"]},
                               cfg["controls"], detector)["prompt"] for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(device)
            sampling = ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p,
                         "num_return_sequences": samples} if samples > 1 else {"do_sample": False})
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 **sampling)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, r in enumerate(batch):
                cands = [(t, parse_completion(t, len(r["lines"])))
                         for t in texts[j * samples:(j + 1) * samples]]
                if selector is not None:
                    best, _ = selector.pick(r["lines"], cands)
                else:
                    best = next((k for k, (_, ls) in enumerate(cands) if ls is not None), None)
                lines = cands[best][1] if best is not None else None
                valid += lines is not None
                f.write(json.dumps({
                    "id": r["id"], "model": f"sft:{args.adapter}", "base": cfg["base"],
                    "style": "sft", "prompt_version": "sft", "samples": samples,
                    "rerank": bool(args.rerank),
                    "raw": cands[best if best is not None else 0][0],
                    "rewrites": {str(k): l for k, (l, o) in enumerate(zip(lines, r["lines"]))
                                 if l != o} if lines else {},
                    "lines": lines, "valid": lines is not None,
                }, ensure_ascii=False) + "\n")
            print(f"{min(i + args.batch_size, len(records))}/{len(records)}", flush=True)
    print(f"valid (right number of lines): {valid}/{len(records)}, output: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the LoRA rewriter.")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="name of a radio_ready.sft output")
    t.add_argument("--sources", nargs="*", default=None,
                   help="pair sources to train on (paradehate, genius, kidzbop); default all")
    t.add_argument("--name", required=True, help="output folder under outputs/sft_models")
    t.add_argument("--resume", default=None, help="adapter folder to continue from")
    t.add_argument("--base", default=BASE_MODEL)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=8, help="micro-batch")
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--max-len", type=int, default=1024)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--max-val", type=int, default=500)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--limit", type=int, default=None, help="training examples (smoke test)")
    t.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("predict")
    p.add_argument("--adapter", required=True, help="folder under outputs/sft_models")
    p.add_argument("--input", default=str(EVAL_SECTIONS_JSONL))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples", type=int, default=1, help=">1: top-p sampling")
    p.add_argument("--rerank", action="store_true",
                   help="choose among the samples with generate.Selector (filters and ranking)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.manual_seed(getattr(args, "seed", 42))
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
