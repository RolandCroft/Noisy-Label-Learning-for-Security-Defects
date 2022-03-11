import torch
from transformers import AutoTokenizer, AutoModel

# Pytorch and conda environment clash workaround
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CodeBERT:
    """
    CodeBERT class.
    Tokenize and infer programming language embeddings.

    Example:
    cb = CodeBert()
    sentences = ["int myfunciscool(float b) { return 1; }", "int main()"]
    embeddings = cb.encode(sentences)
    """

    def __init__(self, download=False):
        """Initialise a Tokenizer and Embedding Model."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load from cache for speedup or to avoid internet regulations
        cb_path = 'environments/codeBERT/'
        cbc_path = 'environments/codeBERT_cache/'
        if not download:
            self.tokenizer = AutoTokenizer.from_pretrained(cb_path)
            self.model = AutoModel.from_pretrained(cb_path)
        else:
            os.mkdir(cb_path)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base",
                                                           cache_dir=cbc_path)
            self.model = AutoModel.from_pretrained("microsoft/codebert-base",
                                                   cache_dir=cbc_path)
            self.tokenizer.save_pretrained(cb_path)
            self.model.save_pretrained(cb_path)

        self.model.to(self.device)
        print('CodeBERT Initialized.')

    def encode(self, sents: list):
        """Get CodeBert embeddings from a list of sentences."""

        tokens = [self.tokenizer.sep_token + " " + i for i in sents]
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer(tokens, **tk_args).to(self.device)
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"])[1]


if __name__ == '__main__':
    cb = CodeBERT(download=True)
    # sentences = ["int myfunciscool(float b) { return 1; }", "int main()"]
    # embeddings = cb.encode(sentences)
    # print(embeddings.shape)
