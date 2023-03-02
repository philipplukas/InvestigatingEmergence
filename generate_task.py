import itertools
import random

"""
Create synthetic data according to the  
Composing-Table-Lookup Task described in https://arxiv.org/abs/1802.06467
"""

FUNC_DOMAIN = ["0", "1"]
DOMAIN_SIZE = 3
NUM_TASKS = 8

FUNCTION_SYMBOLS = "abcdefgh"
assert(len(FUNCTION_SYMBOLS) == NUM_TASKS)

KEYS = list(itertools.product(FUNC_DOMAIN, repeat=DOMAIN_SIZE))
ALL_TASKS = list(itertools.permutations(KEYS, len(FUNC_DOMAIN)**DOMAIN_SIZE))

def generate_tasks():
    tasks = {}
    samples = random.sample(ALL_TASKS, NUM_TASKS)
    for idx, symbol in enumerate(FUNCTION_SYMBOLS):
        tasks[symbol] = dict(zip(KEYS, samples[idx]))

    return tasks


def generate_composite_tasks(base_tasks):
    base_tasks = base_tasks.items()
    base_tasks_prods = itertools.product(base_tasks, repeat=2)
    comp_tasks = {}
    for prod in base_tasks_prods:
        name = prod[0][0] + prod[1][0]


        outputs = []
        for key in KEYS:
            intermediate = prod[0][1][key]
            final = prod[1][1][intermediate]
            outputs.append(intermediate + final)
        comp_tasks[name] = dict(zip(KEYS, outputs))

    return comp_tasks


def infinite_samples(base_tasks, comp_tasks):

    base_task_prefix = "NC"
    comp_task_prefix = "PC"

    while True:
        rand  = random.random()
        choose_base_task = True
        if rand > 1/8:
            choose_base_task = False

        if choose_base_task:
            prefix = base_task_prefix
            choice = random.choice(list(base_tasks.keys()))
            key = KEYS[random.randint(0,7)]
            task_in = key
            task_out = base_tasks[choice][key]

        else:
            prefix = comp_task_prefix
            choice = random.choice(list(comp_tasks.keys()))
            key = KEYS[random.randint(0,7)]
            task_in = key
            task_out = comp_tasks[choice][key]

        complete = prefix + choice + ":" + "".join(task_in) + ". " + "".join(task_out) + "."
        yield complete

    
if __name__ == "__main__":
    print(KEYS)

    base_tasks = generate_tasks()
    comp_tasks = generate_composite_tasks(base_tasks)

    #print(base_tasks)
    #print(comp_tasks['aa'])

    iterator = infinite_samples(base_tasks, comp_tasks)
    for i in range(100):
        print(next(iterator))





