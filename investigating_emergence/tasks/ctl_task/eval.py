import sys
import os
import pickle

if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))


from transformer_xl.pytorch.mem_transformer import MemTransformerLM
from ..ctl_task.task import CTLTask
from torchtext.transforms import VocabTransform, ToTensor

import torch

from typing import List

"""
Evaluate language model on specifc task.
"""
class Evaluator:
    
    def __init__(self, model, vocab, device, target_len, max_depth) -> None:
        #model = torch.load("LM-TFM-ctl/20230313-191014/model.pt", map_location=torch.device('cpu'))
        #model.to(torch.device(device))
        #model.eval()
        #print(model)

        # Make sure, model is in eval mode

        self.valid_filename = "/cluster/home/guphilip/SemesterProject/InvestigatingEmergence/investigating_emergence/data/ctl/valid.txt"
        self.model = model
        self.vocab = vocab
        self.device = device
        self.target_len = target_len
        self.max_depth = max_depth

        # Load basetasks
        with open('/cluster/home/guphilip/SemesterProject/InvestigatingEmergence/investigating_emergence/data/ctl/base_tasks.pickle', 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
            self.base_tasks = pickle.load(f)

        #self.vocab_transform = VocabTransform(self.vocab)
        self.to_tensor_transform = ToTensor()


    def tokenize(self, raw: str) -> List[str]:
        return list(raw)
    

    def preprocess(self, tokenized: str) -> torch.Tensor:

        #encoded =  self.vocab_transform(tokenized)
        tensor = self.to_tensor_transform(list(map(ord,tokenized)))
        device_tensor = tensor.to(device=self.device)
        return device_tensor

    def predict_next_token(self, sentence):
        encoded = self.preprocess(self.tokenize(sentence))

        hidden = self.model._forward(encoded.reshape(-1,1))[0][-self.target_len:]
       
        logit = self.model.crit._compute_logit(hidden, self.model.crit.out_layers[0].weight,
                                        self.model.crit.out_layers[0].bias, self.model.crit.out_projs[0])

        logit_one = logit[:,-1,:]

        #print(self.vocab.lookup_tokens(torch.argmax(logit, 2).reshape(-1).tolist()))
        #print(list(map(chr, (torch.argmax(logit[:,-1,:], 1).reshape(-1).tolist()))))

        max_idx = torch.argmax(logit_one,dim=1)

        #return self.vocab.lookup_token(max_idx)
        return chr(max_idx[-1])

    def predict_next_n_token(self, sentence, n):

        out = ""
        current_sentence = sentence
        for idx in range(n):
            next_sym = self.predict_next_token(current_sentence)

            current_sentence = current_sentence + next_sym
            out += next_sym

        return out

    def calculate_accuracy(self):

        correct = [0] * self.max_depth
        total = [0] * self.max_depth

        task = CTLTask(self.base_tasks)

        #with open(self.valid_filename, 'r') as fp:
        #   for idx, line in enumerate(fp.readlines()):
        for depth in range(1,self.max_depth+1):
                #if idx % 100 == 0:
                #       continue

            iterator = task.infinite_samples(self.base_tasks, depth=depth, eval_mode=True)
            for i in range(5):

                #line = line.rstrip()
                # Not interested in mask part
                line = next(iterator)[0]
                input_symbol = line[1]

                # output_symbol = int(input_symbol)
                # for i in range(depth):
                #    output_symbol = self.base_tasks[line[i+2]][(output_symbol,)][0]
                # output_symbol = str(output_symbol)

                query = line[:-2]
                answer = line[-2]
                # print("Predicting next tokens")
                pred_answer = self.predict_next_token(query)

                if i == 0:
                    print("Query {}".format(query))
                    print("Predicted Answer: {}".format(pred_answer))
                    print("Answer {}".format(answer))
                    print()

                if pred_answer == answer:
                    correct[depth-1] += 1
                total[depth-1] += 1


        for idx, el in enumerate(total):
            if el == 0:
                total[idx] = 1
        return [corr/tot for (corr, tot) in zip(correct, total)]

if __name__ == "__main__":
    os.chdir(os.path.dirname(sys.argv[0]))
    sys.path.append('transformer_xl/pytorch/utils')
    evaluate = Evaluator(device="cpu")
    score = evaluate.calculate_accuracy('../../data/ctl/valid.txt')
    print("score: {}".format(score))

    #0.6345
    #0.644