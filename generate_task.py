import itertools
import random

"""
Create synthetic data according to the  
Composing-Table-Lookup Task described in https://arxiv.org/abs/1802.06467
"""
class CTLTask():

    def __init__(self):
      
        # User numbers instad of binary ["0", "1"] since they are more general
        self.func_domain = range(2)   
        self.domain_size = 3
        self.num_tasks = 8
        self.max_depth = 2

        self.function_symbols = "abcdefgh"
        assert(len(self.function_symbols) == self.num_tasks)

        self.keys = list(itertools.product(self.func_domain, repeat=self.domain_size))

        # Don't generate all tasks at once because this takes too long.
        self.all_tasks = list(itertools.permutations(self.keys, len(self.func_domain)**self.domain_size))

    def generate_tasks(self):
        tasks = {}
        samples = random.sample(self.all_tasks, self.num_tasks)
        for idx, symbol in enumerate(self.function_symbols):
            tasks[symbol] = dict(zip(self.keys, samples[idx]))

        return tasks


    def infinite_samples(self, base_tasks):

        base_task_prefix = "NC"
        comp_task_prefix = "PC"

        
        while True:
            rand_depth  = random.randint(1,self.max_depth+1)

            if rand_depth > 1:
                prefix = comp_task_prefix
            else:
                prefix = base_task_prefix
        
            task_out =  ""
            choice = random.choice(list(base_tasks.keys()))
            key = self.keys[random.randint(0,7)]
            task_in = key
            task_out_step = base_tasks[choice][key]

            for step in range(rand_depth):
                
                task_out = task_out + "".join(map(str,task_out_step))
                choice = random.choice(list(base_tasks.keys()))
                task_out_step = base_tasks[choice][task_out_step]


            complete = prefix + choice + ":" + "".join(map(str, task_in)) + ". " + "".join(task_out) + "."
            yield complete

        
if __name__ == "__main__":

    task = CTLTask()

    base_tasks = task.generate_tasks()

    #print(base_tasks)
    #print(comp_tasks['aa'])

    iterator = task.infinite_samples(base_tasks)
    for i in range(100):
        print(next(iterator))





