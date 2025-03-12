from transformers import AutoTokenizer

def check_special_token(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer.special_tokens_map)
    print("<think>" in tokenizer.get_vocab())
    print("</think>" in tokenizer.get_vocab())
    print("<answer>" in tokenizer.get_vocab())
    print("</answer>" in tokenizer.get_vocab())
    print(tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False))
    print("<|Assistant|>" in tokenizer.get_vocab())

check_special_token("Qwen/Qwen2.5-0.5B-Instruct")

check_special_token("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")