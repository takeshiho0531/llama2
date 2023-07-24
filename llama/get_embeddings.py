import torch  # type: ignore
from model import ModelArgs, Transformer  # type: ignore
from tokenizer import Tokenizer  # type: ignore

# モデルの引数を設定
model_args = ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=10000)

# モデルの初期化
model = Transformer(model_args)
model.load_state_dict(torch.load("path_to_pretrained_model.pth"))  # TODO

# SentencePieceモデルをロード
tokenizer = Tokenizer("path_to_sentencepiece_model.model")  # TODO

# 特定の文字列をトークン化して、トークンIDに変換する
text = "Hello, how are you?"  # TODO?
tokens = tokenizer.encode(text, bos=True, eos=True)

# トークンIDをPyTorchテンソルに変換
tokens_tensor = torch.tensor(tokens).unsqueeze(0)

with torch.no_grad():
    model_output = model(tokens_tensor, start_pos=0)
    embedding = model_output[0]  # 最初のサンプル（1つのテキスト）のembeddingを取得


print("元のテキスト:", text)
print("embeddingのサイズ:", embedding.size())
print("embedding:", embedding)
