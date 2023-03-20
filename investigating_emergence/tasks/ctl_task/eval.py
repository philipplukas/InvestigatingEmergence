import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))


from transformer_xl.mem_transformer import MemTransformerLM
from transformer_xl.data_utils import get_lm_corpus
import torch

"""
Evaluate language model on specifc task.
"""
class Evaluator:
    
    def __init__(self, device="gpu") -> None:
        model = torch.load("LM-TFM-ctl/20230313-191014/model.pt", map_location=torch.device('cpu'))
        model.to(torch.device(device))
        model.eval()
        #print(model)

        self.model = model
        self.corpus = get_lm_corpus('../../data/ctl', 'ctl')
        self.vocab = self.corpus.vocab
        self.device = device

    def tokenize(self, sentence):
        return self.vocab.tokenize(sentence)
    
    def encode(self, tokenized):

        encoded = self.vocab.convert_to_tensor(tokenized)
        encoded = encoded.to(device=self.device)
 
        return encoded

    def predict_next_token(self, sentence):
        encoded = self.encode(self.tokenize(sentence))

        hidden = self.model._forward(encoded.reshape(-1,1))[0]
       
        logit = self.model.crit._compute_logit(hidden, self.model.crit.out_layers[0].weight,
                                        self.model.crit.out_layers[0].bias, self.model.crit.out_projs[0])

        logit = logit[-1,:,:]

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
                parts = line.split('.')
                query = parts[0] + '.'
                answer = parts[1] + '.'
                #print("Predicting next tokens")
                pred_answer = self.predict_next_n_token(query, 3)
                print("Query {}".format(query))
                print("Predicted Answer: {}".format(pred_answer))
                print("Answer {}".format(answer))
                print()
                if pred_answer == answer:
                    correct += 1
                total += 1

        return correct/total

if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))
    sys.path.append('transformer_xl/pytorch/utils')
    evaluate = Evaluator(device="cpu")
    score = evaluate.calculate_accuracy('../../data/ctl/valid.txt')
    print("score: {}".format(score))

    #0.6345
    #0.644