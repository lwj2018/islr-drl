import torch
import torch.nn.functional as F
import time
from utils.metricUtils import *
from utils import AverageMeter

def train_isolated(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Set trainning mode
    model.train()

    end = time.time()
    for i, data in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs and labels
        mat, target = data
        mat = mat.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(mat)

        # compute the loss
        loss = criterion(outputs, target)

        # backward & optimize
        loss.backward()
        optimizer.step()

        # compute the metrics
        prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update average value
        losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())

        if i==0 or i % log_interval == log_interval-1:
            info = ('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'
                    .format(
                        epoch, i, len(trainloader), batch_time=batch_time,
                        data_time=data_time, loss=losses,  top1=top1, top5=top5,
                        lr=optimizer.param_groups[-1]['lr']))
            print(info)
            writer.add_scalar('train loss',
                    losses.avg,
                    epoch * len(trainloader) + i)
            writer.add_scalar('train acc',
                    top1.avg,
                    epoch * len(trainloader) + i)
            # Reset average meters 
            losses.reset()
            top1.reset()
            top5.reset()
