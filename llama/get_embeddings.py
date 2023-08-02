from .example_get_embeddings import Llama

def main(prompts: list[str]):
    max_seq_len=128
    max_batch_size=4
    max_gen_len: int = 64

    # ex. prompts=["Hello, how are you?"]
    generator = Llama.build(
        ckpt_dir="/app/weight/llama-2-7b/",
        tokenizer_path="/app/weight/tokenizer.model",
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("build!")

    embedding_list= generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=0.6,
        top_p=0.9,
    )

    print("done!")
    return embedding_list