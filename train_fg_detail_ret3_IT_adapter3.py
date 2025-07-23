# encoding:utf-8
# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import os, random, copy
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click
from random import randint
import utils
from dataset import data_ret3
import engine_detail_fgw_ret3_IT

from model.c2cmodel import ClipSelector, TokenizerSelector, C2cmodel

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_AMFMN.yaml', type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()
    print("Now Run config file is ", opt.path_opt)
    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    return options


def main(options):

    seed = 2008
    torch.manual_seed(seed)
    #np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])

    # Create dataset, model, criterion and optimizer
    train_loader, val_loaders = data_ret3.get_ITDWloaders(options,test_data_index=[0,1,2])

    if not isinstance(val_loaders, list):
        val_loaders = [val_loaders]
    longclippath = options['model']['longclippath']
    tokenizername = options['model']['clipname']
    clipmodel = ClipSelector(tokenizername, longclippath,adapter_type="DC")
    tokenizer = TokenizerSelector(tokenizername)
    model = C2cmodel(tokenizer=tokenizer, clipmodel=clipmodel).to('cuda')
 
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters())
                                    , lr=options['optim']['lr'], betas=(0.9, 0.98))
    print('Model has {} parameters'.format(utils.params_count(model)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=options['optim']['lr_update_epoch'],
                                                gamma=options['optim']['lr_decay_param'])
    warm_up = options['optim']['warmup']
    warm_up_lr = 0.1 * options['optim']['lr']
    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']

            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
            for val_loader in val_loaders:
                rsum, all_scores =  engine_detail_fgw_ret3_IT.validate(val_loader, model)
                print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
            start_epoch = 0
    else:
        start_epoch = 0

    # Train the Model
    best_rsum = 0
    best_score = ""
    # rsum, all_scores = engine_onlytest.validate(val_loader, model,tokenizername)
    for epoch in range(start_epoch, options['optim']['epochs']):

        if epoch < warm_up: 
            print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            scheduler.step()
            print("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        engine_detail_fgw_ret3_IT.train(train_loader, model, optimizer, epoch, opt=options)
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': options,
                'Eiters': model.Eiters,
            },
            False,
            filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'], epoch, best_rsum),
            prefix=options['logs']['ckpt_save_path'],
            model_name=options['model']['name']
        )
        # evaluate on validation set
        if epoch % options['logs']['eval_step'] == 0:
            val_loaders_num = len(val_loaders)
            list_rsum = []
            list_all_scores = []
            for i in range(val_loaders_num):
                rsum, all_scores = engine_detail_fgw_ret3_IT.validate(val_loaders[i], model)
                list_rsum.append(rsum)
                list_all_scores.append(all_scores)
            rsum = sum(list_rsum)
            all_scores = list_all_scores
            is_best = rsum > best_rsum
            if is_best:
                best_scores = all_scores
            best_rsum = max(rsum, best_rsum)

            # save ckpt
            utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': 'baseline',
                    'model': model.state_dict(),
                    'best_rsum': best_rsum,
                    'options': options,
                    'Eiters': model.Eiters,
                    'optimizer': optimizer.state_dict(),
                    #'scaler': loss_scaler.state_dict()
                },
                is_best,
                filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'], epoch, best_rsum),
                prefix=options['logs']['ckpt_save_path'],
                model_name=options['model']['name']
            )

            print("Current {}th fold.".format(options['k_fold']['current_num']))
            print("Now  score:")
            for all_score in all_scores:
                print(all_score)
            print("Best score:")
            for best_score in best_scores:
                print(best_score)
            for all_score,best_score in zip(all_scores,best_scores):
                utils.log_to_txt(
                    contexts= "Epoch:{} ".format(epoch+1) + all_score,
                    filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
                )
                utils.log_to_txt(
                    contexts= "Best:   " + best_score,
                    filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
                )


def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options


if __name__ == '__main__':
    options = parser_options()

    # print(options[])
    # make logger
    tb_logger.configure(options['logs']['logger_name'], flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # k_fold verify
    for k in range(options['k_fold']['nums']):
        print("=========================================")

        # update save path
        update_options = update_options_savepath(options, k) 

        # run experiment
        main(update_options)
