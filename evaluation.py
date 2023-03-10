
import transformer_xl.pytorch
from transformer_xl.pytorch.mem_transformer import MemTransformerLM
from transformer_xl.pytorch.data_utils import get_lm_corpus
import torch
import sys


"""
Evaluate language model on specifc task.
"""
class Evaluator:
    
    def __init__(self, device="gpu") -> None:
        model = torch.load("transformer-xl/pytorch/LM-TFM-ctl/20230307-110317/model.pt")
        model.to(torch.device(device))
        model.eval()

        self.model = model
        self.corpus = get_lm_corpus('data/ctl', 'ctl')
        self.vocab = self.corpus.vocab
        self.device = device

    def tokenize(self, sentence):
        return self.vocab.tokenize(sentence)
    
    def encode(self, tokenized):

        encoded = self.vocab.encode_sents(tokenized)
        encoded = encoded[0].to(device=self.device)

    def predict_next_token(self, sentence):
        encoded = self.encode(self.tokenize(sentence))
        hidden = self.model._forward(encoded[0].reshape(1,-1))[0]

        logit = self.model.crit._compute_logit(hidden, self.model.crit.out_layers[0].weight,
                                        self.model.crit.out_layers[0].bias, self.model.crit.out_projs[0])

        max_idx = torch.argmax(logit)
        return self.vocab.idx2sym[max_idx]

    def predict_next_n_token(self, sentence, n):

        out = ""
        current_sentence = sentence
        for idx in range(n):
            next_sym = self.predict_next_token(current_sentence)
            current_sentence = current_sentence + next_sym
            out += next_sym

        return out

    def calculate_accuracy(self, val_file_name):
        correct = 0
        total = 0

        with open(val_file_name, 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                parts = line.split('. ')
                query = parts[0]
                answer = parts[1]
                pred_answer = self.predict_next_n_token(query, len(answer))
                if pred_answer == answer:
                    correct += 1
                total += 1

        return correct/total

if __name__ == "__main__":
    sys.path.append('transformer_xl/pytorch/utils')
    evaluate = Evaluator(device="gpu")
    score = evaluate.calculate_accuracy('data/ctl/valid.txt')
    print("score: {}".format(score))
