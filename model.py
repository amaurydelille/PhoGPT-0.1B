import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import time
import csv
from safetensors.torch import save_file

project_root = Path(__file__).parent.resolve()

loss_metrics_file = project_root / "metrics" / "loss.csv"
loss_metrics_file.parent.mkdir(parents=True, exist_ok=True)
if not loss_metrics_file.exists() or loss_metrics_file.stat().st_size == 0:
    with open(loss_metrics_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'batch', 'loss'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIM_MODEL = 512
DIM_FF_HIDDEN = 2048 
NUM_LAYERS = 6  
NUM_HEADS = 8
SRC_VOCAB_SIZE = 32000
TGT_VOCAB_SIZE = 32000

def get_sentences(file_path: str) -> List[str]:
    with open(file=file_path, mode='r', encoding='utf-8') as file:
        return file.readlines()

class AttentionUtils:
    @staticmethod
    def make_kv_pad_mask(tokens: torch.Tensor, pad_id: Optional[int] = None) -> Optional[torch.Tensor]:
        if not pad_id:
            return None

        keep = (tokens != pad_id)
        return keep.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)

    @staticmethod
    def make_q_pad_mask(tokens: torch.Tensor, pad_id: Optional[int]) -> Optional[torch.Tensor]:
        if not pad_id:
            return None

        return (tokens != pad_id).unsqueeze(1).unsqueeze(3).to(dtype=torch.bool)

class Tokenizer:
    def __init__(self, max_len: int, spm_model_path: str) -> None:
        self.max_len = max_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        self.pad_token = 0
        self.bos_token = self.sp.bos_id() if self.sp.bos_id() != -1 else self.sp.vocab_size()
        self.eos_token = self.sp.eos_id() if self.sp.eos_id() != -1 else self.sp.vocab_size() + 1

        base_vocab = self.sp.vocab_size()
        extra = 0
        if self.sp.bos_id() == -1:
            extra += 1
        if self.sp.eos_id() == -1:
            extra += 1
        self.vocab_size = base_vocab + extra

        logger.info(f"SPM tokens: PAD={self.pad_token}, BOS={self.bos_token}, EOS={self.eos_token}, Vocab Size={self.vocab_size}")

    def _pad_encoded_content(self, ids: List[int]) -> torch.Tensor:
        if len(ids) >= self.max_len:
            return torch.tensor(ids[:self.max_len], dtype=torch.long)
        return torch.tensor(ids + [self.pad_token] * (self.max_len - len(ids)), dtype=torch.long)

    def encode(self, sentence: str) -> torch.Tensor:
        ids = self.sp.encode(sentence, out_type=int, add_eos=True)
        return self._pad_encoded_content(ids)

    def encode_batch(self, corpus: List[str]) -> torch.Tensor:
        batch_ids = self.sp.encode(corpus, out_type=int, add_eos=True)
        return torch.stack([self._pad_encoded_content(ids) for ids in batch_ids])

    def decode(self, encoded: torch.Tensor) -> str:
        ids = [i for i in encoded.flatten().tolist() if i != self.pad_token and i not in (self.bos_token, self.eos_token)]
        return self.sp.decode(ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask: bool = False) -> None:
        assert d_model % num_heads == 0, "Wrong shapes buddy"

        super().__init__()
        self.mask = mask
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.k_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.v_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)

        self.out_proj = nn.Linear(in_features=self.d_k * num_heads, out_features=self.d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2) # B, n, N, d_k

        attention_scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5) # B, d_k, N, N

        if self.mask:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device))
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

        if attention_mask is not None:
            if attention_mask.shape[-2] == 1:  # kv-only mask -> expand to (B,1,N,N)
                attention_mask = attention_mask.expand(B, 1, N, N)
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))


        attention_values = F.softmax(attention_scores, dim=-1) @ V # B, d_k, N, d_k
        attention_values = attention_values.transpose(1, 2).contiguous().view(B, N, self.d_k * self.num_heads)

        return self.out_proj(attention_values), K, V

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        assert d_model % num_heads == 0, "Wrong shapes buddy"

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.k_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.v_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)

        self.out_proj = nn.Linear(in_features=self.d_k * num_heads, out_features=self.d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, kv_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Cross-attention: Q from decoder, K/V from encoder output.
        
        Args:
            x: decoder hidden states (B, N_q, d_model)
            encoder_output: encoder output (B, N_k, d_model)
            kv_pad_mask: mask for encoder padding (B, 1, 1, N_k)
        """
        B, N_q, D = x.shape
        N_k = encoder_output.size(1)

        Q = self.q_proj(x).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(encoder_output).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(encoder_output).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, heads, N_q, N_k)

        if kv_pad_mask is not None:
            attention_scores = attention_scores.masked_fill(~kv_pad_mask.expand(B, 1, N_q, N_k), float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_values = attention_weights @ V  # (B, heads, N_q, d_k)
        attention_values = attention_values.transpose(1, 2).contiguous().view(B, N_q, self.d_k * self.num_heads)

        return self.out_proj(attention_values)

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_hat = (x - mean) / (std + self.eps)

        return self.gamma * x_hat + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, mask=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_hidden=d_ff, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_mha, _, _ = self.mha.forward(x=x, attention_mask=pad_mask)
        x = self.norm1(x + self.dropout1(x_mha))
        x_ff = self.ff(x)
        x = self.norm2(x + x_ff)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, mask=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(d_model=d_model)
        self.cmha = CrossMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
        self.ff = FeedForward(d_model=d_model, d_hidden=d_ff, dropout=dropout)
        self.norm3 = LayerNorm(d_model=d_model)

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            kv_pad_mask: Optional[torch.Tensor] = None,
            q_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_mha, _, _ = self.mha.forward(x=x, attention_mask=q_pad_mask)
        x = self.norm1(x + self.dropout1(x_mha))
        x_cmha = self.cmha.forward(x, encoder_output=encoder_output, kv_pad_mask=kv_pad_mask)
        x = self.norm2(x + self.dropout2(x_cmha))
        x_ff = self.ff(x)
        x = self.norm3(x + x_ff)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            kv_pad_mask: Optional[torch.Tensor] = None,
            q_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output=encoder_output, kv_pad_mask=kv_pad_mask, q_pad_mask=q_pad_mask)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, 
                 src_vocab_size: int, tgt_vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, vocab_size=tgt_vocab_size, dropout=dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) as per original paper
        scale = self.d_model ** 0.5
        src_embedded = self.dropout(self.pe(self.src_embedding(src.long()) * scale))  # (B, seq_len, d_model)
        tgt_embedded = self.dropout(self.pe(self.tgt_embedding(tgt.long()) * scale))  # (B, seq_len, d_model)

        src_kv_mask = AttentionUtils.make_kv_pad_mask(tokens=src, pad_id=0)
        tgt_q_mask = AttentionUtils.make_q_pad_mask(tokens=tgt, pad_id=0)

        encoder_output = self.encoder(src_embedded, pad_mask=src_kv_mask)
        return self.decoder(x=tgt_embedded, encoder_output=encoder_output, kv_pad_mask=src_kv_mask, q_pad_mask=tgt_q_mask)  # (B, seq_len, vocab_size)

class Trainer:
    def __init__(self, batch_size: int, epochs: int, device: str, verbose: bool) -> None:
        self.d_model = DIM_MODEL
        self.num_heads = NUM_HEADS
        self.max_len = 128
        self.device = torch.device(device)
        self.verbose = verbose

        self.batch_size = batch_size
        self.epochs = epochs

        en_sents = project_root / "dataset" / "en_sents"
        vi_sents = project_root / "dataset" / "vi_sents"

        self.en_sents = get_sentences(str(en_sents))
        self.vi_sents = get_sentences(str(vi_sents))

        self.en_tokenizer = Tokenizer(max_len=self.max_len, spm_model_path=str(project_root / "tokenizer" / "en_spm.model"))
        self.vi_tokenizer = Tokenizer(max_len=self.max_len, spm_model_path=str(project_root / "tokenizer" / "vi_spm.model"))

        vocab_size = max(self.en_tokenizer.vocab_size, self.vi_tokenizer.vocab_size)
        self.vocab_size = int(vocab_size)

        self.transformer = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=NUM_LAYERS,
            d_ff=DIM_FF_HIDDEN,
            src_vocab_size=self.en_tokenizer.vocab_size,
            tgt_vocab_size=self.vi_tokenizer.vocab_size,
        )

        logger.info(f"Loaded transformer with: {sum(p.numel() for p in self.transformer.parameters())} parameters")

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vi_tokenizer.pad_token)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
        self.transformer.to(self.device)

    def get_batch(self, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = min(start_idx + self.batch_size, len(self.en_sents))

        en_batch = self.en_sents[start_idx:end_idx]
        vi_batch = self.vi_sents[start_idx:end_idx]

        en_tokens = self.en_tokenizer.encode_batch(en_batch)  # (B, seq_len) token IDs
        vi_tokens = self.vi_tokenizer.encode_batch(vi_batch)  # (B, seq_len) token IDs

        if self.verbose:
            logger.info(f"Source batch: {en_batch}")
            logger.info(f"Tokenized source batch: {en_tokens}")

        return en_tokens.to(self.device), vi_tokens.to(self.device)

    def train_epoch(self, epoch: int) -> float:
        self.transformer.train()
        total_loss = 0.0
        num_batches = (len(self.en_sents) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            src, tgt_full = self.get_batch(batch_idx * self.batch_size)  # (B, L=128)

            bos = torch.full((tgt_full.size(0), 1), self.vi_tokenizer.bos_token, dtype=torch.long, device=self.device)
            tgt_in = torch.cat([bos, tgt_full[:, :-1]], dim=1)  # (B, L)

            self.optimizer.zero_grad()
            logits = self.transformer(src, tgt_in)  # (B, L_logit, tgt_vocab)

            B, L_logit, V_tgt = logits.shape
            B2, L_tgt = tgt_full.shape
            assert B == B2, f"Batch mismatch: logits B={B}, tgt B={B2}"
            if L_logit != L_tgt:
                tgt_full = tgt_full[:, :L_logit]

            loss = self.criterion(logits.reshape(-1, V_tgt), tgt_full.reshape(-1).long())

            with open(loss_metrics_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1, loss.item()])

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        return total_loss / max(1, num_batches)

    def train(self):
      checkpoint_dir = project_root / "checkpoints"
      checkpoint_dir.mkdir(parents=True, exist_ok=True)
      last_epoch = 0
      try:
        start_time = time.time()
        for epoch in range(self.epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Average Loss: {avg_loss:.4f}")

            state = {
                "state_dict": self.transformer.state_dict(),
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_layers": NUM_LAYERS,
                "d_ff": DIM_FF_HIDDEN,
                "max_len": self.max_len,
            }
            base = checkpoint_dir / f"transformer_epoch_{epoch + 1}"
            torch.save(state, str(base.with_suffix(".pth")))
            torch.save(state, str(base.with_suffix(".bin")))
            save_file(state["state_dict"], str(base.with_suffix(".safetensors")))
            last_epoch = epoch + 1

        end_time = time.time()
        logger.info(f"Training time: {end_time - start_time:.2f} seconds")
      except KeyboardInterrupt as e:
          state = {
              "state_dict": self.transformer.state_dict(),
              "vocab_size": self.vocab_size,
              "d_model": self.d_model,
              "num_heads": self.num_heads,
              "num_layers": NUM_LAYERS,
              "d_ff": DIM_FF_HIDDEN,
              "max_len": self.max_len,
          }
          base = checkpoint_dir / f"transformer_latest_epoch_{last_epoch}"
          torch.save(state, str(base.with_suffix(".pth")))
          torch.save(state, str(base.with_suffix(".bin")))
          save_file(state["state_dict"], str(base.with_suffix(".safetensors")))
          raise e
      except Exception as e:
          state = {
              "state_dict": self.transformer.state_dict(),
              "vocab_size": self.vocab_size,
              "d_model": self.d_model,
              "num_heads": self.num_heads,
              "num_layers": NUM_LAYERS,
              "d_ff": DIM_FF_HIDDEN,
              "max_len": self.max_len,
          }
          base = checkpoint_dir / f"transformer_latest_epoch_{last_epoch}"
          torch.save(state, str(base.with_suffix(".pth")))
          torch.save(state, str(base.with_suffix(".bin")))
          save_file(state["state_dict"], str(base.with_suffix(".safetensors")))
          raise e


class Translator:
    def __init__(self, model_path: str) -> None:
        ckpt = torch.load(model_path, map_location='mps')
        self.model = Transformer(
            d_model=ckpt.get("d_model", DIM_MODEL),
            num_heads=ckpt.get("num_heads", NUM_HEADS),
            num_layers=ckpt.get("num_layers", NUM_LAYERS),
            d_ff=ckpt.get("d_ff", DIM_FF_HIDDEN),
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.en_tokenizer = Tokenizer(max_len=ckpt.get("max_len", 128), spm_model_path=str(project_root / "tokenizer" / "en_spm.model"))
        self.vi_tokenizer = Tokenizer(max_len=ckpt.get("max_len", 128), spm_model_path=str(project_root / "tokenizer" / "vi_spm.model"))

    def translate(self, sentence: str) -> str:
        src = self.en_tokenizer.encode(sentence).unsqueeze(0)
        bos_id = self.vi_tokenizer.bos_token
        eos_id = self.vi_tokenizer.eos_token
        generated = [bos_id]
        max_len = self.vi_tokenizer.max_len

        device = next(self.model.parameters()).device
        src = src.to(device)
        
        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt = torch.tensor(generated, dtype=torch.long, device=src.device).unsqueeze(0)
                logits = self.model(src, tgt)
                next_id = logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_id)
                if next_id == eos_id:
                    break
        out_ids = []
        for tid in generated[1:]:
            if tid == eos_id:
                break
            out_ids.append(tid)
        return self.vi_tokenizer.decode(torch.tensor(out_ids, dtype=torch.long))

    def translate_stream(self, sentence: str) -> None:
        src = self.en_tokenizer.encode(sentence).unsqueeze(0)
        bos_id = self.vi_tokenizer.bos_token
        eos_id = self.vi_tokenizer.eos_token
        generated = [bos_id]
        max_len = self.vi_tokenizer.max_len

        device = next(self.model.parameters()).device
        src = src.to(device)
        
        previous_text = ""
        
        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt = torch.tensor(generated, dtype=torch.long, device=src.device).unsqueeze(0)
                logits = self.model(src, tgt)
                next_id = logits[:, -1, :].argmax(dim=-1).item()
                generated.append(next_id)
                
                if next_id == eos_id:
                    break
                
                out_ids = [tid for tid in generated[1:] if tid != eos_id and tid != self.vi_tokenizer.pad_token]
                if out_ids:
                    current_text = self.vi_tokenizer.decode(torch.tensor(out_ids, dtype=torch.long))
                    
                    if len(current_text) > len(previous_text):
                        new_text = current_text[len(previous_text):]
                        print(new_text, end="", flush=True)
                        previous_text = current_text
                # time.sleep(0.1)
        
        print() 

    def run(self):
        while True:
            sentence = input("You:    ")
            if sentence.lower() == "exit":
                break
            print("PhoGPT: ", end="")
            self.translate_stream(sentence)
            print()

if __name__ == "__main__":

    translator = Translator(model_path=str(project_root / "checkpoints" / "transformer_epoch_1.pth"))
    original = "How are you ?"
    translator.run()
