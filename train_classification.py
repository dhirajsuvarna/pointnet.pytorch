from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, DMUDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='saved_models', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed) #later: 

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'dmunet':
    dataset = DMUDataset(
        dataPath=opt.dataset,
        classification=True,
        npoints=opt.num_points,
        split = 'train')

    test_dataset = DMUDataset(
        dataPath=opt.dataset,
        classification=True,
        npoints=opt.num_points,
        split = 'test',
        data_augmentation=False)

else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True)

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True)

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda() #debug: comment this while debugging on cpu


tb = SummaryWriter() # create SummaryWriter object for Tensorboard
#tb.add_graph(classifier) # adding network to tensorboard

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda() #comment if debugging on cpu
        optimizer.zero_grad()
        classifier = classifier.train()
        with torch.autograd.detect_anomaly():
            pred, trans, trans_feat = classifier(points)
            train_loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                train_loss += feature_transform_regularizer(trans_feat) * 0.001
            # print(f"f3 Weights Min: {classifier.fc3.weight.min()}")
            # print(f"f3 Weights Max: {classifier.fc3.weight.max()}")
            
            train_loss.backward()
            
            # print(f"f3 Gradients Min: {classifier.fc3.weight.grad.min()}")
            # print(f"f3 Gradients Max: {classifier.fc3.weight.grad.max()}")

            optimizer.step()

        train_loss += train_loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        train_acc = correct.item() / float(opt.batchSize)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, train_loss.item(), train_acc))
        print(f"Predictions: {torch.argmax(pred, dim=1)}")
        print(f"Targets: {target}")
        print(f"NLL-Loss {train_loss}")

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            test_loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            test_acc = correct.item()/float(opt.batchSize)
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), test_loss.item(), test_acc))
            print(f"Predictions: {torch.argmax(pred, dim=1)}")
            print(f"Targets: {target}")
            print(f"NLL-Loss {test_loss}")

            # Write to Tensorboard
            tb.add_scalar("Training Loss", train_loss.item(), epoch) # here the last term should not be epoch
            tb.add_scalar("Training Accuracy", train_acc, epoch) # here the last term should not be epoch
            tb.add_scalar("Test Loss", test_loss.item(), epoch) # here the last term should not be epoch
            tb.add_scalar("Test Accuracy", test_acc, epoch) # here the last term should not be epoch

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

tb.close()


total_correct = 0
total_testset = 0
all_preds = torch.tensor(data=[],dtype=torch.long).cuda()
all_targets = torch.tensor(data=[], dtype=torch.long).cuda()
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    all_preds = torch.cat((all_preds, torch.argmax(pred, dim=1)), dim=0)
    all_targets = torch.cat((all_targets, target))
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print(f"Shape of all_preds: {all_preds.shape}")
print(f"Shape of all_targets: {all_targets.shape}")
print("final accuracy {}".format(total_correct / float(total_testset)))

################################
# Generate Confusion Matrix
################################
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plot_cm

classNames = [ k for k, v in dataset.classes.items() ]
all_targets = all_targets.cpu().numpy()
all_preds = all_preds.cpu().numpy()
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(len(classNames),len(classNames)))
plot_cm.plot_confusion_matrix(cm, classNames)