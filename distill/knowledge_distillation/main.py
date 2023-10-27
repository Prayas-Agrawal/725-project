import os
import time
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

def train(student, dataloader, optimizer, scheduler, loss_calculator, device, args, teacher=None, val_dataloader=None):
    best_accuracy = 0
    best_epoch = 0

    if teacher is not None:
        teacher.eval()

    for epoch in range(1, args.epoch+1):
        # train one epoch
        train_step(student, dataloader, optimizer, loss_calculator, device, args, epoch, teacher)

        # validate the network
        if (val_dataloader is not None) and (epoch % args.valid_interval == 0):
            accuracy = measure_accuracy(student, val_dataloader, device)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

        # learning rate schenduling
        if scheduler: scheduler.step()

        # save check point
        if (epoch % args.save_epoch == 0) or (epoch == args.epoch):
            torch.save({'argument': args,
                        'epoch': epoch,
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler':  scheduler.state_dict(),
                        'loss_log': loss_calculator.loss_log},
                        os.path.join(args.save_path, 'check_point_%d.pth'%epoch))

    print("Finished Training, Best Accuracy: %f (at %d epochs)"%(best_accuracy, best_epoch))
    return student

def train_step(student, dataloader, optimizer, loss_calculator, device, args, epoch, teacher=None):
    student.train()

    for i, (inputs, labels) in enumerate(dataloader, 1):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(inputs.to(device))

        teacher_outputs = None
        if teacher is not None and args.distillation_weight > 0.0:
            with torch.no_grad():
                teacher_outputs = teacher(inputs.to(device))

        loss = loss_calculator(outputs          = outputs,
                               labels           = labels.to(device),
                               teacher_outputs  = teacher_outputs)
        loss.backward()
        optimizer.step()

        # print log
        if i % args.print_interval == 0:
            print("%s: Epoch [%3d/%3d], Iteration [%5d/%5d], Loss [%s]"%(time.ctime(),
                                                                         epoch,
                                                                         args.epoch,
                                                                         i,
                                                                         len(dataloader),
                                                                         loss_calculator.get_log()))
    return None

def measure_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().cpu().item()

    print("Accuracy of the network on the 10000 test images: %f %%"%(100 * correct / total))

    return correct / total


def get_transforms(train_flag, args):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # if args.data in ['CIFAR10']:
    #     # Mean and Std of CIFAR10 Train data
    #     MEAN = [0.4914, 0.4822, 0.4465]
    #     STD  = [0.2471, 0.2435, 0.2616]

    #     if train_flag:
    #         transformer = transforms.Compose([
    #                     transforms.RandomCrop(32, padding=4),
    #                     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(MEAN, STD)])
    #     else:
    #         transformer = transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(MEAN, STD)])

    return transform

def get_dataloader(train_flag, args):
    transformer = get_transforms(train_flag, args)

    dataset = torchvision.datasets.__dict__[args.data](root         = args.data_path,
                                                       train        = train_flag,
                                                       download     = True,
                                                       transform    = transformer
                                                       )

    dataloader = torch.utils.data.DataLoader(dataset        = dataset,
                                             batch_size     = args.batch_size,
                                             shuffle        = train_flag == True,
                                             num_workers    = args.num_workers,
                                             pin_memory     = True)
    return dataloader

class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight):
        super().__init__()

        self.temperature         = temperature
        self.distillation_weight = distillation_weight
        self.loss_log            = defaultdict(list)
        self.kldiv               = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs=None):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        Reference: https://github.com/peterliht/knowledge-distillation-pytorch
        """

        # Distillation Loss
        soft_target_loss = 0.0
        if teacher_outputs is not None and self.distillation_weight > 0.0:
            soft_target_loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1), F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.temperature ** 2)

        # Ground Truth Loss
        hard_target_loss = F.cross_entropy(outputs, labels, reduction='mean')

        total_loss = (soft_target_loss * self.distillation_weight) + hard_target_loss

        # Logging
        if self.distillation_weight > 0:
            self.loss_log['soft-target_loss'].append(soft_target_loss.item())

        if self.distillation_weight < 1:
            self.loss_log['hard-target_loss'].append(hard_target_loss.item())

        self.loss_log['total_loss'].append(total_loss.item())

        return total_loss

    def get_log(self, length=100):
        log = []
        # calucate the average value from lastest N losses
        for key in self.loss_log.keys():
            if len(self.loss_log[key]) < length:
                length = len(self.loss_log[key])
            log.append("%s: %2.3f"%(key, sum(self.loss_log[key][-length:]) / length))
        return ", ".join(log)


def get_optimizer(models, args):
    lr = 0.1
    weight_decay = 5e-4
    adam_betas = (0.9, 0.999)
    scheduler = None
    lr_milestones = [150,225]
    lr_stepsize = 150
    lr_gamma = 0.1
    optim_ = "Adam"

    # if args.optim == 'SGD':
    #     optimizer = optim.SGD(models.parameters(),
    #                           lr        = args.lr,
    #                           momentum  = args.sgd_momentum,
    #                           weight_decay=args.weight_decay)

    if optim_ == 'Adam':
        optimizer = optim.Adam(models.parameters(),
                               lr           = lr,
                            #    betas        = adam_betas,
                            #    weight_decay = weight_decay
                               )
    if scheduler == None:
        pass

    elif scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer     = optimizer,
                                              step_size     = lr_stepsize,
                                              gamma         = lr_gamma)

    elif scheduler == 'MStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer     = optimizer,
                                                   milestones    = lr_milestones,
                                                   gamma         = lr_gamma)


    return optimizer, scheduler


def Distillation(args, device):
    # load argument

    # make a network instance
    network = load_network(args.model, args.model_load, args.num_class, device)

    # validation data loader
    val_loader = get_dataloader(train_flag = False, args = args)

    if args.train_flag:
        optimizer, scheduler = get_optimizer(network, args)

        # load a teacher for knowledge distillation
        teacher = None
        if args.teacher_load:
            teacher = load_network(args.teacher, args.teacher_load, args.num_class, device)

        # train data loader
        train_loader = get_dataloader(train_flag = True, args = args)

        # make loss calculator
        loss_calculator = LossCalulcator(args.temperature, args.distillation_weight).to(device)

        # train the network
        print("Training the network...")
        network = train(student         = network,
                        dataloader      = train_loader,
                        optimizer       = optimizer,
                        scheduler       = scheduler,
                        loss_calculator = loss_calculator,
                        device          = device,
                        args            = args,
                        teacher         = teacher,
                        val_dataloader  = val_loader)

    else:
        # evaluate the network
        print("Evalute the network...")
        measure_accuracy(network, val_loader, device)
