import tiktoken

class Utils():
    def __init__(self, cfg):
        self.encoding = tiktoken.get_encoding(cfg['TIKTOKEN_ENCODING_NAME'])

    def count_tokens(self, string):
        return len(self.encoding.encode(string))