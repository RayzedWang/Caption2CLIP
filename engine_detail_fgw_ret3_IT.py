#encoding:utf-8
# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import time
from torch.autograd import profiler
import clip
import torch
import numpy as np
import sys
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm
from model import longclip
import random

def adjust_learning_rate(optimizer, warmup_steps, current_step, base_lr,total_steps=40*4000, start_lr=2e-7):
    if current_step < warmup_steps:
        lr = start_lr + (base_lr - start_lr) * (current_step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(train_loader, model, optimizer, epoch, opt={},stage=True):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']
    warmupepoch = opt['optim']['warmup']
    maxepoch = opt['optim']['epochs']
    if stage==True:
        base_lr = opt['optim']['lr']
    else:
        base_lr = 4e-4
    warmup_steps = warmupepoch * len(train_loader)
    accumulation_steps = 2
    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    USE_AMP = True
    scaler = GradScaler(enabled=True)
    optimizer.zero_grad(set_to_none=True)
    for i, train_data in enumerate(train_loader):
        images, captions, lengths, ids, detail,word,label = train_data
        if epoch < warmupepoch:
            current_step = epoch * len(train_loader) + i
            adjust_learning_rate(optimizer, warmup_steps, current_step, base_lr,total_steps=maxepoch*len(train_loader))

        batch_size = images.size(0)
        margin = float(margin)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)

        if torch.cuda.is_available():
            input_visual = input_visual.cuda()

        with autocast(): 
            drop_cross = random.random()
            scores,_,detail_feature,cap_feature = model(input_visual, text=captions,detail_text=detail,word=word,device='cuda',is_train=True,stage=stage)

            loss1 = utils.calcul_loss(scores, input_visual.size(0), margin, max_violation=max_violation, labels=label,is_sum=False)

            if stage==True:
                loss2 = utils.calcul_Aligin_loss(detail_feature=detail_feature,text_feature=cap_feature,margin=0.075,is_sum=False)

                loss = loss1 + loss2
            else:
                loss = loss1
            
        scaler.scale(loss).backward()
        if torch.isnan(loss):
            print(f"Loss is NaN at iteration {i}, skipping this step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if (i + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            train_logger.update('L', loss.cpu().data.numpy())

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) 


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)

    if (i + 1) % accumulation_steps != 0:
        if grad_clip > 0:
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        train_logger.update('L', loss.cpu().data.numpy())

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) 

def validate(val_loader, model,stage=True):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))
    input_text = ['']*len(val_loader.dataset)
    input_detail = ['']*len(val_loader.dataset)
    input_word = ['']*len(val_loader.dataset)
    input_text_lengeth = [0]*len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids, detail, words,label = val_data
        for (id, img, cap, key, l, det,word) in zip(ids, (images.numpy().copy()), (captions), images , lengths, (detail), words):
            input_visual[id] = img
            input_text[id]=cap
            input_detail[id]=det
            input_word[id]=word
            input_text_lengeth[id] = l


    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
    input_detail = [input_detail[i] for i in range(0,len(input_detail),5)]
    input_word = [input_word[i] for i in range(0,len(input_word),5)]


    d = utils.shard_dis_Detail_FGW_Version(input_visual, input_text, input_detail, model, words=input_word, lengths=input_text_lengeth,stage=stage )

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
  
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score


def validate_test(val_loader, model,tokenizer='longclip'):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 224, 224))

    input_text = ['']*len(val_loader.dataset)
    input_detail = ['']*len(val_loader.dataset)
    input_word = ['']*len(val_loader.dataset)
    input_text_lengeth = [0] * len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids, detail, words = val_data
        if tokenizer == 'longclip':
            captions = longclip.tokenize(captions)
            detail = longclip.tokenize(detail)
            words = longclip.tokenize(words)
        else:
            captions = clip.tokenize(captions)
            detail = clip.tokenize(detail)
            words = longclip.tokenize(words)
        for (id, img, cap, key, l, det, word) in zip(ids, (images.numpy().copy()), (captions), images , lengths, (detail), words):
            input_visual[id] = img
            input_text[id] = cap
            input_detail[id] = det
            input_word[id]=word
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis_Detail_FGW_Version(input_visual, input_text, input_detail, model, words=input_word, lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d
