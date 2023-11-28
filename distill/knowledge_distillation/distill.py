from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Model_Student(nn.Module):
    def __init__(self):
        super(Model_Student, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class Model_CNN(nn.Module):
    def __init__(self):
        super(Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

from collections import defaultdict
class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight, lossMode="simple", forget=None, teacher=None, device = None):
        super().__init__()
        self.lossMode = lossMode
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.loss_log = defaultdict(list)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.forget = forget
        self.teacher = teacher
        self.device = device

    def simpleDistillLoss(self, outputs, teacher_outputs):
        loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1), F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.temperature ** 2)
        return loss
    
    def influenceFunctionLoss(self, outputs, teacher_outputs):
        if(self.lossMode != "influence" or self.teacher is None or self.forget is None or self.device is None):
            # print("calculating FROM SIMPLE")
            return self.simpleDistillLoss(outputs, teacher_outputs)
        # print("calculating from influence")
        forget_influence = 0
        for batch_idx, (data, target) in enumerate(self.forget):
            data, target = data.to(self.device), target.to(self.device)
            output = self.teacher(data)
            loss = F.nll_loss(output, target)
            forget_influence += loss

        loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1), F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.temperature ** 2)
        return loss - forget_influence

    def forward(self, outputs, labels, teacher_outputs=None):
        # Distillation Loss
        soft_target_loss = 0.0
        # print("teacher outputs", teacher_outputs, self.distillation_weight)
        if teacher_outputs is not None and self.distillation_weight > 0.0:
            if(self.lossMode == "simple"):
                soft_target_loss = self.simpleDistillLoss(outputs, teacher_outputs)
            else:
                soft_target_loss = self.influenceFunctionLoss(outputs, teacher_outputs)

        # Ground Truth Loss
        hard_target_loss = F.nll_loss(outputs, labels)

        total_loss = (soft_target_loss * self.distillation_weight) + hard_target_loss

        # # Logging
        if self.distillation_weight > 0:
            self.loss_log['soft-target_loss'].append(soft_target_loss)

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


def train(args, model, device, train_loader, optimizer, epoch, teacher=None, lossMode="simple", forget=None):
    model.train()
    loss_calculator = LossCalulcator(temperature, distillation_weight, lossMode=lossMode, forget=forget, teacher=teacher, device = device).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        teacher_outputs = None
        # print("train teacher", teacher, distillation_weight)
        if teacher is not None and distillation_weight > 0.0:
            with torch.no_grad():
                # print("getting teacher outputs")
                teacher_outputs = teacher(data.to(device))
                # print("teacher outputs", teacher_outputs)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_calculator(outputs          = output,
                               labels           = target.to(device),
                               teacher_outputs  = teacher_outputs)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, LossLogs: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_calculator.get_log()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

class DistArgs:
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 1.0
    gamma = 0.7
    no_cuda = False
    no_mps = False
    dry_run = False
    # seed = 1
    log_interval = 10
    save_model = False


def Distill(saveFile, teacher=None, lossMode="simple", forgetDataCount=None, trainData=None, testData=None):
    # Training settings
    args = DistArgs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    forget_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = trainData
    dataset2 = testData
    train_loader =  torch.utils.data.DataLoader(dataset1,**train_kwargs) if train_loader else None
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    random_sampler = torch.utils.data.RandomSampler(dataset1, num_samples=forgetDataCount)
    forget_loader = None
    if(forgetDataCount is not None):
        forget_loader = torch.utils.data.DataLoader(dataset1, sampler=random_sampler, **forget_kwargs)
    modelKind = Model_CNN
    if(teacher is not None):
        print("Training with student model")
        modelKind = Model_Student
        # train_loader = forget_loader
    model = modelKind().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, teacher=teacher,  lossMode=lossMode, forget=forget_loader)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), saveFile + ".pt")
    return model
