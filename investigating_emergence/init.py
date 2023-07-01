import argparse
import torch
import os
import numpy as np
import time

import torch.nn as nn
import torch.optim as optim

from transformer_xl.pytorch.mem_transformer import MemTransformerLM
from transformer_xl.pytorch.utils.exp_utils import create_exp_dir
from transformer_xl.pytorch.utils.data_parallel import BalancedDataParallel

from torch.utils.data import DataLoader

from tasks.ctl_task.dataset import CTLDataset
from tasks.enwik_task.dataset import EnwikDataset
from tasks.mixed_task.dataset import MixedDataset

from itertools import cycle

BASE_PATH = os.path.dirname(__file__)

def init():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='ctl',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8', 'ctl', 'mixed'],
                        help='dataset name')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=3,
                        help='number of heads')
    parser.add_argument('--d_head', type=int, default=3,
                        help='head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    parser.add_argument('--d_inner', type=int, default=128,
                        help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.1,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--scheduler', default='constant', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    parser.add_argument('--tgt_len', type=int, default=127,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=127,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=4000,
                        help='evaluation interval')
    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al.')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--sample_softmax', type=int, default=-1,
                        help='number of samples in sampled softmax')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--finetune_v2', action='store_true',
                        help='finetune v2')
    parser.add_argument('--finetune_v3', action='store_true',
                        help='finetune v3')
    parser.add_argument('--fp16', action='store_true',
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can '
                        'improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument'
                        ' supersedes --static-loss-scale.')
    
    # My own arguments not included in transformer-x;
    parser.add_argument('--mixing-rate',  type=float, default=1, 
                        help='Percentage of data coming from ctl-task, between 0 and 1')
    parser.add_argument('--use-mask-training',  action="store_true", 
                        help='Percentage of data coming from ctl-task, between 0 and 1')
    args = parser.parse_args()
    args.tied = not args.not_tied


    # Inner width to twice the model width
    args.d_inner = 4*args.d_model

    # Adapt head dimension accoding to model dimension and number of heads
    args.d_head = args.d_model // args.n_head

    # DEBUG
    # args.batch_size = 1

    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    #logging = create_exp_dir(args.work_dir,
    #    scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)
    logging = create_exp_dir(args.work_dir, debug=args.debug)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
            args.fp16 = False
        else:
            try:
                from apex.fp16_utils import FP16_Optimizer
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                args.fp16 = False

    device = torch.device('cuda' if args.cuda else 'cpu')

    ###############################################################################
    # Load data
    ###############################################################################

    #corpus = get_lm_corpus(args.data, args.dataset)
    #ntokens = len(corpus.vocab)
    #args.n_token = ntokens

    eval_batch_size = 10
    # tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    #     device=device, ext_len=args.ext_len)
    # va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    #     device=device, ext_len=args.ext_len)
    # te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    #     device=device, ext_len=args.ext_len)

    if args.dataset == "ctl":
        train_data = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "train", args.tgt_len, device, eval_mode=True, all_chars=True)
        valid_data = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "valid", args.eval_tgt_len, device, eval_mode=True, all_chars=True)
        test_data = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "test", args.eval_tgt_len, device, eval_mode=True, all_chars=True)

    elif args.dataset == "enwik8":
        train_data = EnwikDataset(os.path.join(BASE_PATH,"data/enwik8/"), "train", args.tgt_len, device)
        valid_data = EnwikDataset(os.path.join(BASE_PATH,"data/enwik8/"), "valid", args.eval_tgt_len, device)
        test_data = EnwikDataset(os.path.join(BASE_PATH, "data/enwik8/"), "test", args.eval_tgt_len, device)

    elif args.dataset == "mixed":
        train_data_ctl = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "train", args.tgt_len, device, all_chars=True, eval_mode=True)
        valid_data_ctl = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "valid", args.eval_tgt_len, device, all_chars=True, eval_mode=True)
        test_data_ctl = CTLDataset(os.path.join(BASE_PATH, "data/ctl/"), "test", args.eval_tgt_len, device, all_chars=True, eval_mode=True)

        train_data_enwik = EnwikDataset(os.path.join(BASE_PATH, "data/enwik8/"), "train", args.tgt_len, device)
        valid_data_enwik = EnwikDataset(os.path.join(BASE_PATH, "data/enwik8/"), "valid", args.eval_tgt_len, device)
        test_data_enwik = EnwikDataset(os.path.join(BASE_PATH, "data/enwik8/"), "test", args.eval_tgt_len, device)

        train_data = MixedDataset(train_data_ctl, train_data_enwik, args.batch_size, args.mixing_rate, device)
        valid_data = valid_data_ctl #MixedDataset(valid_data_ctl, valid_data_enwik, args.batch_size, 0.9, device)
        test_data = test_data_ctl #MixedDataset(test_data_ctl, test_data_enwik, args.batch_size, 0.9, device)

    mixed_data = train_data

    if args.dataset == "mixed":
        # This data already comes in batched from
        tr_iter = DataLoader(train_data, batch_size=None)
        va_iter = DataLoader(valid_data) #, batch_size=None)
        te_iter = DataLoader(test_data) #, batch_size=None)
        enwik8_iter = DataLoader(EnwikDataset(os.path.join(BASE_PATH, "data/enwik8/"), "valid", args.eval_tgt_len, device))

    else: 
        # This data already comes in batched from
        tr_iter = DataLoader(train_data, args.batch_size)
        va_iter = DataLoader(valid_data, args.batch_size)
        te_iter = DataLoader(test_data, args.batch_size)

    vocab = train_data.get_vocab()
    ntokens = len(train_data.get_vocab())
    args.n_token = ntokens

    logging("Data set loaded")

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    ###############################################################################
    # Build the model
    ###############################################################################
    def init_weight(weight, d_layer_input=args.d_model):
        if args.pre_lnorm:
            nn.init.normal_(weight, 0.0, np.sqrt(2 / (args.n_layer * d_layer_input)))
            return

        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                #if args.pre_lnorm:
                #    nn.init.uniform_(m.weight, -np.sqrt(3/args.d_model), np.sqrt(3/args.d_model))
                #else:
                init_weight(m.weight, args.d_model)
            if hasattr(m, 'bias') and m.bias is not None:
                #if args.pre_lnorm:
                #    nn.init.uniform_(m.bias, -np.sqrt(3/args.d_model), np.sqrt(3/args.d_model))
                #else:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        if args.pre_lnorm:
                            nn.init.kaiming_normal_(m.emb_projs[i], mode="fan_in", nonlinearity="linear")
                        else:
                            nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                if args.pre_lnorm:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                else:
                    init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        if args.pre_lnorm:
                            init_weight(m.out_projs[i])
                            #nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
                        else:
                            nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                if args.pre_lnorm:
                    nn.init.normal_(m.weight, 1.0, args.init_std)
                    #init_weight(m.weight)
                else:
                    nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = args.dropout

    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = args.dropatt

    if args.restart:
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

    if args.fp16:
        model = model.half()

    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                            model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    #### optimizer
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                momentum=args.mom)
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    #### scheduler
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                args.max_step, eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        scheduler = None
        pass

    if args.cuda and args.fp16:
        # If args.dynamic_loss_scale is False, static_loss_scale will be used.
        # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
        optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale = args.static_loss_scale,
                                dynamic_loss_scale = args.dynamic_loss_scale,
                                dynamic_loss_args = {'init_scale': 2 ** 16})

    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')

    if args.dataset == "mixed":
        if scheduler:
            return args, logging, optimizer, None, model, para_model, tr_iter, va_iter, te_iter, enwik8_iter, device, vocab ,scheduler, mixed_data
        else:
            return args, logging, optimizer, None, model, para_model, tr_iter, va_iter, te_iter, enwik8_iter, device, vocab ,None, mixed_data
    else:
        if scheduler:
            return args, logging, optimizer, None, model, para_model, tr_iter, va_iter, te_iter, None, device, vocab, scheduler, mixed_data
        else:
            return args, logging, optimizer, None, model, para_model, tr_iter, va_iter, te_iter, None, device, vocab, None, mixed_data