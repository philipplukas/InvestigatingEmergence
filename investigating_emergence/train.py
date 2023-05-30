# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools
import wandb

import numpy as np
import re

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

#from transformer_xl.data_utils import get_lm_corpus
from transformer_xl.pytorch.mem_transformer import MemTransformerLM
from transformer_xl.pytorch.utils.exp_utils import create_exp_dir
from transformer_xl.pytorch.utils.data_parallel import BalancedDataParallel

from tasks.ctl_task.dataset import CTLDataset
from tasks.enwik_task.dataset import EnwikDataset
from tasks.mixed_task.dataset import MixedDataset
from tasks.ctl_task.eval import Evaluator

from data.statistics import get_data_statistics

from transformer_xl.pytorch.mem_transformer import MemTransformerLM
import init

from itertools import cycle
from itertools import islice

args, logging, optimizer, optimizer_sparse, model, para_model, tr_iter, va_iter, te_iter, enwik8_iter, device, vocab, scheduler = init.init()


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="investigating-emergence",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "dataset": "mixed",
    "function_domain": "range(8)",
    "domain_size": 8,
    "num_tasks": 8,
    "max_depth": 5,
    "n_layer": args.n_layer,
    "d_model": args.d_model,
    "n_head": args.n_head,
    "d_head": args.d_head,
    "d_inner": args.d_inner,
    "dropout": args.dropout,
    "optim": args.optim,
    "tgt_len": args.tgt_len,
    "eval_tgt_len": args.eval_tgt_len,
    "scheduler": args.scheduler,
    "ext_len": 0,
    "mem_len": 0,
    "batch_size": args.batch_size,
    "mixing_rate": args.mixing_rate
    }
)


# Log datset statistics

train_stats = get_data_statistics("train")
valid_stats = get_data_statistics("valid")


# Example from wandb
#data = [[label, val] for (label, val) in train_stats.items()]
#table = wandb.Table(data=data, columns = ["label", "value"])
#wandb.log({"train_data_dist" : wandb.plot.bar(table, "label", "value",
#                               title="train_data_dist")})

#data = [[label, val] for (label, val) in valid_stats.items()]
#table = wandb.Table(data=data, columns = ["label", "value"])
#wandb.log({"valid_data_dist" : wandb.plot.bar(table, "label", "value",
#                               title="valid_data_dist")})

# In case there is an exception, still finish wandb run


#def notify_exception(type, value, tb):
#    wandb.finish()
#sys.excepthook = notify_exception


train_iter = iter(cycle(tr_iter))
va_iter = iter(cycle(va_iter))

if args.dataset == "mixed":
    enwik8_iter = iter(cycle(enwik8_iter))

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    all_predictions = []
    all_mask = []
    all_targets = []
    with torch.no_grad():
        mems = tuple()
        logging("Before eval loop")
        for i, (data, target, seq_len, mask) in enumerate(islice(eval_iter,5)):

            data = data.transpose(0,1).contiguous()
            target = target.transpose(0,1).contiguous()
            #target = target.reshape(1,-1)

            mask = mask.transpose(0,1).contiguous()

            # seq_len should only be one number
            # But since we use automati batching we get duplicates
            # Just extract the first element
            seq_len = data.size()[0] #[0].item()

            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret, hidden = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]

            logits = model.crit._compute_logit(hidden, model.crit.out_layers[0].weight,
                                        model.crit.out_layers[0].bias, model.crit.out_projs[0])


            # Get correct shape, sames as in transformer model code
            # logits = logits.view(-1, logits.size(-1))
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions)
            all_mask.append(mask)
            all_targets.append(target)

            
            # Check if mask is in use
            if args.dataset == "mixed":
                if mask[0][0].item() != -1:
                     loss = (loss*mask).flatten().sum() / mask.flatten().sum()
            elif args.dataset == "ctl":
                if mask[0][0].item() != -1:
                    if mask.flatten().sum().item() == 0:
                        loss = (loss*mask).flatten().sum()
                    else:
                        loss = (loss*mask).flatten().sum() / mask.flatten().sum()
                else:
                    loss = loss.mean() #sum() #.mean()

            else:
                if mask[0].item() != -1:
                    loss = (loss*mask).mean()
                else:
                    loss = loss.mean()
       
            total_loss += loss.float().item()#seq_len * loss.float().item()
            total_len += 1 #seq_len
        logging("After eval loop")

        # Enwik8 loop

        if args.dataset == "mixed":
            enwik_loss = 0
            for  batch, (data, target, seq_len, mask) in enumerate(islice(enwik8_iter,5)):
                data = data.transpose(0,1).contiguous()
                target = target.transpose(0,1).contiguous()

                ret, _ = para_model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss)

                enwik_loss += loss
                #loss = (loss*mask).float().mean().type_as(loss)

                #masked_loss = (loss*mask).float().mean().type_as(loss)
            enwik_loss = enwik_loss / 100


    if args.dataset == "ctl" or args.dataset == "mixed":
        # Compute masked symbols correct.

        """
        masked_indices = torch.cat(all_mask, 0)
        predictions = torch.cat(all_predictions, 0)
        all_targets = torch.cat(all_targets, 0)
        a = torch.argwhere(masked_indices)
        total = len(torch.argwhere(masked_indices))


        correct = total - len(torch.argwhere((predictions-all_targets)*masked_indices))
        correct_rate = correct / total

        # Assume full answers are separated by 0 mask-values
        index_spans = []

        zero_before = False
        current_start_index = -1
        current_idx = 0

        all_mask = torch.flatten(masked_indices)
        all_targets = torch.flatten(all_targets)
        predictions = torch.flatten(predictions)

        logging("Before indices calculation")

        # Calculate indices of continous 1-sequences
        for el in all_mask:
            if el.item() == 1:
                if zero_before:
                    current_start_index = current_idx
                else:
                    pass    

                zero_before = False

            else:
                if current_start_index > -1:
                    index_spans.append( (current_start_index, current_idx) )
                    current_start_index = -1

                zero_before = True
            current_idx += 1

        total_spans = len(index_spans)
        total_correct = 0

        for span in index_spans:
            span_len = span[1] - span[0]

            #print(torch.sum(predictions[span[0]:span[1]] == all_targets[span[0]:span[1]]))
            #print(span_len)
            if torch.sum(predictions[span[0]:span[1]] == all_targets[span[0]:span[1]]).item() == span_len:
                total_correct += 1

            #print("predictions") 
            #print(predictions[span[0]-1:span[1]-1])
            #print(all_targets[span[0]:span[1]])

        print(vocab.lookup_tokens(predictions[:20].tolist()))
        print(vocab.lookup_tokens(all_targets[:20].tolist()))

        total_correct_rate = total_correct / total_spans
        """

        # Alternative computation
        evaluator = Evaluator(model,vocab, device, args.tgt_len, 5)
        total_correct_rate = evaluator.calculate_accuracy()
        correct_rate = total_correct_rate


    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    #logging("End of eval")

    if args.dataset == "mixed":
        return total_loss / total_len, correct_rate, total_correct_rate, enwik_loss
    elif args.dataset == "ctl":
        return total_loss / total_len, correct_rate, total_correct_rate
    else:
        return total_loss / total_len


def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    #train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    #train_iter = iter(tr_iter)

    for batch, (data, target, seq_len, mask) in enumerate(train_iter):

        # Get it  into the shape expected by the transfor-mem code
        data = data.transpose(0,1).contiguous()
        #target = target.reshape(1,-1)
        target = target.transpose(0,1).contiguous()

        #print(list(map(chr, data.squeeze().tolist())))

        if mask[0][0].item() != -1:
            mask = mask.transpose(0,1).contiguous()

        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret, _ = para_model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret, _ = para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            #if train_step % args.log_interval == 0:
            #print(loss[:,0][:16])
            #print(mask[:,0][:16])
            #loss = loss.float().mean().type_as(loss)
            actual_seq_len = seq_len.to(device=device).flatten().sum().float()
            if actual_seq_len == 0:
                loss = (loss*mask).flatten().sum()
            else:
                loss = ((loss*mask).flatten().sum() / actual_seq_len).type_as(loss)

            #masked_loss = (loss*mask).float().mean().type_as(loss)

            wandb.log({"train_cross_entropy": loss, "train_ppl": math.exp(loss)})
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
        #if True:
            #logging("Before calling eval")
            if args.dataset == 'mixed':
                val_loss, correct, total_correct, enwik_loss = evaluate(va_iter)
            elif args.dataset == "ctl":
                val_loss, correct, total_correct = evaluate(va_iter)
            else:
                val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)

            # log metrics to wandb
            if args.dataset == 'mixed':
                wandb.log({"ctl_cross_entropy_val": val_loss, 
                        "ctl_ppl_val": math.exp(val_loss),
                        "enwik8_cross_entropy_val": enwik_loss, 
                        "enwik8_ppl_val": math.exp(enwik_loss)})
                        #"correct answers": total_correct})

                for idx, el in enumerate(correct):
                    wandb.log({"ctl: correct answers depth {}".format(idx+1): el})

            elif args.dataset == "ctl":
                wandb.log({"ctl_cross_entropy_val": val_loss, 
                        "ctl_ppl_val": math.exp(val_loss)})
                        #"correct answers": total_correct})

                for idx, el in enumerate(correct):
                    wandb.log({"ctl: correct answers depth {}".format(idx+1): el})
            else:
                wandb.log({"cross_entropy_val": val_loss, 
                            "ppl_val": math.exp(val_loss)})

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break

# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    logging('Start training')
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')
    wandb.finish()

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)))
else:
    logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))
logging('=' * 100)

# Cleanup wandb 
# wandb.finish()