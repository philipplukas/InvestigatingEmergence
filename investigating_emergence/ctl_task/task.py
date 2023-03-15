import itertools
import random
import pickle
import copy

from torchtext.data import utils

"""
Create synthetic data according to the  
Composing-Table-Lookup Task described in https://arxiv.org/abs/1802.06467
"""
class CTLTask():

    def __init__(self):
      
        # User numbers instad of binary ["0", "1"] since they are more general
        self.func_domain = range(5)   
        self.domain_size = 3
        self.num_tasks = 8
        self.max_depth = 4

        self.function_symbols = "abcdefgh"
        assert(len(self.function_symbols) == self.num_tasks)

        self.keys = list(itertools.product(self.func_domain, repeat=self.domain_size))

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
            print(idx)
            random.shuffle(shuffle_set)
            print(shuffle_set)
            while shuffle_set in occured_outputs:
                #print("Not new")
                random.shuffle(shuffle_set)

            print(shuffle_set)
            tasks[symbol] = dict(zip(self.keys, shuffle_set))
            occured_outputs.append(copy.deepcopy(shuffle_set))
        #samples = random.sample(self.all_tasks, self.num_tasks)
        #for idx, symbol in enumerate(self.function_symbols):
        #    tasks[symbol] = dict(zip(self.keys, samples[idx]))

        return tasks


    def infinite_samples(self, base_tasks, rejection_sampling=False):

        base_task_prefix = "NC"
        comp_task_prefix = "PC"

        seen = set()

        while True:
            rand_depth  = random.randint(1,self.max_depth)

            if rand_depth > 1:
                prefix = comp_task_prefix
            else:
                prefix = base_task_prefix
        
            task_out =  ""
            choice = random.choice(list(base_tasks.keys()))
            key = self.keys[random.randint(0,self.num_tasks-1)]
            task_in = "".join(map(str, key))
            task_out_step = base_tasks[choice][key]

            for step in range(rand_depth-1):
                task_in += "".join(map(str, task_out_step))
                choice = random.choice(list(base_tasks.keys()))
                task_out_step = base_tasks[choice][task_out_step]

            complete = choice + ":" + "".join(map(str, task_in)) + "." + "".join(map(str,task_out_step)) + "."
            
            if rejection_sampling:
                if complete in seen:
                    continue
                else:
                    seen.add(complete)
            
            yield complete

        
if __name__ == "__main__":

    task = CTLTask()
    
    print("Before base task")
    base_tasks = task.generate_tasks()
    print("After base task")
    
    with open('data/ctl/base_tasks.pickle', 'wb') as f:
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
    iterator = task.infinite_samples(base_tasks, rejection_sampling=True)

    print("Start generating")
    with open("data/ctl/master.txt", 'w') as fp:
         for i in range(150000):
             print(i)
             fp.write(next(iterator) + "\n")





