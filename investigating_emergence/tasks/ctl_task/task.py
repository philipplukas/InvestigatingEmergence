import itertools
import random
import pickle
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

from torchtext.data import utils

"""
Create synthetic data according to the  
Composing-Table-Lookup Task described in https://arxiv.org/abs/1802.06467
"""
class CTLTask():

    def __init__(self, base_tasks=None):
      
        # User numbers instad of binary ["0", "1"] since they are more general
        self.func_domain = range(8)   
        self.domain_size = 1
        self.num_tasks = 8
        self.max_depth = 4

        self.function_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[:8]
        self.start_symbol = "^"
        assert(len(self.function_symbols) == self.num_tasks)

        self.keys = list(itertools.product(self.func_domain, repeat=self.domain_size))

        if base_tasks:
            self.base_tasks = base_tasks

        print("Before generating all tasks")
        # Don't generate all tasks at once because this takes too long.
        # self.all_tasks = list(itertools.permutations(self.keys, len(self.func_domain)**self.domain_size))
        print("Generating all tasks")

    def generate_tasks(self):
        tasks = {}
        occured_outputs = []

        shuffle_set = copy.deepcopy(self.keys)

        #def is_new_list(current, occured_so_far):
        #    return all([current != seen for seen in occured_so_far])

        for idx, symbol in enumerate(self.function_symbols):
            #print(idx)
            random.shuffle(shuffle_set)
            #print(shuffle_set)
            while shuffle_set in occured_outputs:
                #print("Not new")
                random.shuffle(shuffle_set)

            #print(shuffle_set)
            tasks[symbol] = dict(zip(self.keys, shuffle_set))
            occured_outputs.append(copy.deepcopy(shuffle_set))
        #samples = random.sample(self.all_tasks, self.num_tasks)
        #for idx, symbol in enumerate(self.function_symbols):
        #    tasks[symbol] = dict(zip(self.keys, samples[idx]))

        return tasks


    def infinite_samples(self, base_tasks, rejection_sampling=False, depth=None):

        base_task_prefix = "NC"
        comp_task_prefix = "PC"

        seen = set()
        

        while True:
            all_choices = []
            rand_depth  = random.randint(1,self.max_depth)

            if depth:
                rand_depth = depth

            if rand_depth > 1:
                prefix = comp_task_prefix
            else:
                prefix = base_task_prefix
        
            task_out =  ""
            choice = random.choice(list(base_tasks.keys()))
            key = self.keys[random.randint(0,(len(self.func_domain)**self.domain_size)-1)]
            task_in = "".join(map(str, key))
            task_out_step = base_tasks[choice][key]
            task_in_between = []
            all_choices.append(choice)

            for step in range(rand_depth-1):
                task_in_between.append("".join(map(str, task_out_step)))
                choice = random.choice(list(base_tasks.keys()))
                task_out_step = base_tasks[choice][task_out_step]
                all_choices.append(choice)

            combo = zip(all_choices[:-1], task_in_between)  
            between_part = [ part1 + ":" + part2 for (part1, part2) in combo] 

            #complete = "".join(map(str, task_in)) +  ":" + choice + ":" + ".".join(map(str,task_in_between))+ "".join(map(str,task_out_step))
            #complete = "".join(map(str, task_in)) +  ":" + ":".join(all_choices) + ":" + "".join(map(str,task_out_step)) + "."
            #mask = len(task_in)*'0' + '0'  + (len(all_choices) + (len(all_choices)-1))*'0' + '0' + len(task_out_step)*'1' + '0'
            #complete = "".join(map(str, task_in)) +  ":" + ":".join(between_part) + ":" + all_choices[-1] + ":" + "".join(map(str,task_out_step)) + "."
            #mask = len(task_in)*'0' + '0'  + (len(":".join(between_part)))*'0' + '000' + len(task_out_step)*'1' + '0'
            
            # Third version, no : in the middle
            complete = self.start_symbol + "".join(map(str, task_in)) + "".join(all_choices) + ":" + "".join(map(str,task_out_step)) + "."
            mask = '0' + len(task_in)*'0' + (len(all_choices))*'0' + '0' + len(task_out_step)*'1' + '0'
            #print(complete)
            #print(mask)
            assert len(complete) == len(mask)

            if rejection_sampling:
                if complete in seen:
                    continue
                else:
                    seen.add(complete)
            
            yield complete, mask

        
if __name__ == "__main__":

    task = CTLTask()
    
    print("Before base task")
    base_tasks = task.generate_tasks()
    print(base_tasks)
    print("After base task")
    
    with open('investigating_emergence/data/ctl/base_tasks.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(base_tasks, f)


    #print(base_tasks)
    #print(comp_tasks['aa'])

    
    # Generatign data without the use of rejection sampling
    # iterator = task.infinite_samples(base_tasks)

    # with open("data/ctl/train.txt", 'w') as fp:
    #     for i in range(1000000):
    #         fp.write(next(iterator) + "\n")

    # with open("data/ctl/valid.txt", 'w') as fp:
    #     for i in range(2000):
    #         fp.write(next(iterator) + "\n")

    # with open("data/ctl/test.txt", 'w') as fp:
    #     for i in range(2000):
    #         fp.write(next(iterator) + "\n")
    print("Before infinite samples")
    iterator = task.infinite_samples(base_tasks, rejection_sampling=False)

    print("Start generating")
    with open("investigating_emergence/data/ctl/master.txt", 'w') as fp:
         with open("investigating_emergence/data/ctl/master_mask.txt", 'w') as fp2:
            for i in range(100000):
                if i % 10000 == 0:
                    print(i)
                segment, mask = next(iterator)
                fp.write(segment + "\n")
                fp2.write(mask + "\n")

    # Create vocabulary
    all_symbols = list(task.func_domain) + list(task.function_symbols) + [':', '.', '^'] 
    with open("investigating_emergence/data/ctl/vocab.txt", "w") as fp:
        for symbol in all_symbols:
            fp.write(str(symbol) + "\n")

    # Split into train/validation/test
    file_path = "investigating_emergence/data/ctl/master.txt"
    file_path_mask = "investigating_emergence/data/ctl/master_mask.txt"
    df = pd.read_csv(file_path, sep='\n', encoding='utf8', dtype=str, keep_default_na=False, na_values='', names=['task'])
    df2 = pd.read_csv(file_path_mask, sep='\n', encoding='utf8', dtype=str, keep_default_na=False, na_values='', names=['mask'])
    common_df = df.merge(df2, left_index=True, right_index=True)

    train, test = train_test_split(common_df, test_size=0.1)
    train['task'].to_csv("investigating_emergence/data/ctl/train.txt", sep='\n', encoding='utf8', 
                header = False, index = False)
    test['task'].to_csv("investigating_emergence/data/ctl/valid.txt", sep='\n', encoding='utf8', 
                header = False, index = False)
    
    train['mask'].to_csv("investigating_emergence/data/ctl/train_mask.txt", sep='\n', encoding='utf8', 
                header = False, index = False)
    test['mask'].to_csv("investigating_emergence/data/ctl/valid_mask.txt", sep='\n', encoding='utf8', 
                header = False, index = False)

    # Keep test and validation the same
    shutil.copyfile("investigating_emergence/data/ctl/valid.txt", "investigating_emergence/data/ctl/test.txt")
    shutil.copyfile("investigating_emergence/data/ctl/valid_mask.txt", "investigating_emergence/data/ctl/test_mask.txt")


