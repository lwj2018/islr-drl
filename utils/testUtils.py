import os
import torch
import numpy
import time
from utils.metricUtils import *
from utils import AverageMeter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_isolated(model, criterion, testloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Set eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs and labels
            mat, target = data
            mat = mat.to(device)
            target = target.to(device)

            # forward
            outputs = model(mat)

            # compute the loss
            loss = criterion(outputs, target)

            # compute the metrics
            prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update average value
            losses.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())
            if i % 50 == 0:
                print("%d/%d %.2f"%(i,len(testloader),top1.avg))

        info = ('[Test] Epoch: [{0}] [len: {1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Batch Prec@1 {top1.avg:.4f}\t'
                'Batch Prec@5 {top5.avg:.4f}\t'
                .format(
                    epoch, len(testloader), batch_time=batch_time, loss=losses,
                    data_time=data_time,  top1=top1, top5=top5
                    ))
        print(info)
        writer.add_scalar('val acc',
            top1.avg,
            epoch)

    return top1.avg
