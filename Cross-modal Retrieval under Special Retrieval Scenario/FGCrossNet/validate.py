import sys
import torch
from torch.autograd import Variable

def validate(loader, model, args, flag):
    model.eval()
    if args.gpu is not None:
        model = model.module

    total_output = []
    total_label = []
    start_model = True
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
        target = target.cuda(async=True)
        if(flag):
            output = model.forward_txt(input_var)
        else:
            output = model.forward_share(input_var)
        if start_model:
            total_output = output.data.float()
            total_label = target.data.float()
            start_model = False
        else:
            total_output = torch.cat((total_output, output.data.float()), 0)
            total_label = torch.cat((total_label, target.data.float()), 0)

    _, predict = torch.max(total_output, 1)

    acc = torch.sum(torch.squeeze(predict.float() == total_label)).item() / float(total_label.size()[0])
    print('Prec@1:' + str(acc))

    return acc