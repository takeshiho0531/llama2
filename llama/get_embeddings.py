import torch  # type: ignore
from model import ModelArgs  # type: ignore
from tokenizer import Tokenizer  # type: ignore
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
)
import os
import sys
import json

max_seq_len=128
max_batch_size=4

# モデルの引数を設定
with open("/app/weight/llama-2-7b/params.json", "r") as f:
    params = json.loads(f.read())
params=ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params,)
vocab_size = params.vocab_size
tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

# モデルの初期化
# model = Transformer(params)
# model.load_state_dict(torch.load("path_to_pretrained_model.pth"))  # TODO

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
if not model_parallel_is_initialized():
    # if model_parallel_size is None:
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

torch.manual_seed(1)

if local_rank > 0:
    sys.stdout = open(os.devnull, "w")



# SentencePieceモデルをロード
tokenizer = Tokenizer("/app/weight/tokenizer.model")  # TODO

# 特定の文字列をトークン化して、トークンIDに変換する
text = "Hello, how are you?"  # TODO?
#tokens = tokenizer.encode(text, bos=True, eos=True)
prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in text]
print("prompt_tokens:", prompt_tokens)

bsz = len(prompt_tokens)
min_prompt_len = min(len(t) for t in prompt_tokens)
max_prompt_len = max(len(t) for t in prompt_tokens)
assert max_prompt_len <= params.max_seq_len

max_gen_len: int = 64 # TODO?
total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

pad_id = tokenizer.pad_id
tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

prev_pos = 0
eos_reached = torch.tensor([False] * bsz, device="cuda")
input_text_mask = tokens != pad_id

print("min_prompt_len, total_len:", min_prompt_len, total_len)
for cur_pos in range(min_prompt_len, total_len):
    h = tok_embeddings(tokens[:, prev_pos:cur_pos])
    print("type(h):", type(h))
    print("h:", h)




# トークンIDをPyTorchテンソルに変換
# tokens_tensor = torch.tensor(tokens).unsqueeze(0)

# with torch.no_grad():
    # model_output = model(tokens_tensor, start_pos=0)
    # embedding = model_output[0]  # 最初のサンプル（1つのテキスト）のembeddingを取得


print("元のテキスト:", text)
#print("embeddingのサイズ:", embedding.size())
#print("embedding:", embedding)