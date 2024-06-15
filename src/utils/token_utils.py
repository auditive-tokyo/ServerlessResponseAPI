import tiktoken

# OpenAIのモデルに対応するトークナイザを取得
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens_with_tiktoken(text):
    return len(enc.encode(text))

def trim_to_tokens(text, max_tokens):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    else:
        trimmed_tokens = tokens[:max_tokens]
        return enc.decode(trimmed_tokens)