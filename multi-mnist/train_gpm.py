import os
import sys

import argparse
import logging
import pickle
import yaml

from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model_lenet import LenetModel
from model_resnet import ResnetModel
from utils import setup_seed, MinNormSolver
from bypass_bn import enable_running_stats, disable_running_stats
from mtl import PCGrad, CAGrad
# import model_resnet_pretrained



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dset",
    default="multi_fashion_and_mnist",
    type=str,
    help="Dataset for training.",
)

parser.add_argument(
    "--method",
    type=str,
    choices=["mgda", "pcgrad", "a-mgda", "a-pcgrad", "cagrad", "a-cagrad", "VBD"],
    help="MTL weight method",
)

parser.add_argument(
    "--batch_size",
    default=256,
    type=int,
    help="Batch size.",
)

parser.add_argument(
    "--lr", default=1e-3, type=float, help="The initial learning rate for SGD."
)

parser.add_argument(
    "--n_epochs",
    default=100,
    type=int,
    help="Total number of training epochs to perform.",
)

parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")

parser.add_argument(
    "--adaptive",
    default=False,
    type=str2bool,
    help="True if you want to use the Adaptive SAM.",
)

parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")

parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

args.output_dir = "outputs/" + str(args).replace(", ", "/").replace("'", "").replace(
    "(", ""
).replace(")", "").replace("Namespace", "")

print("Output directory:", args.output_dir)
os.system("rm -rf " + args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
    yaml.dump(vars(args), outfile, default_flow_style=False)

log_file = os.path.join(args.output_dir, "MOO-VBD.log")

logging.basicConfig(
    filename=f"./{args.output_dir}/{args.dset}.log",
    level=logging.DEBUG,
    filemode="w",
    datefmt="%H:%M:%S",
    format="%(asctime)s :: %(levelname)-8s \n%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

setup_seed(args.seed)

with open(f"./data/{args.dset}.pickle", "rb") as f:
    trainX, trainLabel, testX, testLabel = pickle.load(f)
trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set = torch.utils.data.TensorDataset(testX, testLabel)

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=args.batch_size, shuffle=False
)
logging.info("==>>> total trainning batch number: {}".format(len(train_loader)))
logging.info("==>>> total testing batch number: {}".format(len(test_loader)))


criterion = nn.CrossEntropyLoss()
model = ResnetModel(2).cuda()
# model1 = model_resnet_pretrained.ResnetModel(2).cuda()

param_amount = 0
for p in model.named_parameters():
    param_amount += p[1].numel()
    print(p[0], p[1].numel())
logging.info(f"total param amount: {param_amount}")

shared_optimizer = torch.optim.Adam(
    model.get_shared_parameters(),
    lr=args.lr,  # momentum=0.9
)

classifier_optimizer = torch.optim.Adam(
    model.get_classifier_parameters(),
    lr=args.lr,  # momentum=0.9
)

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def train(epoch, feature_mat):
    all_losses_1 = 0
    all_losses_2 = 0
    for (it, batch) in tqdm(
        enumerate(train_loader),
        desc=f"Training on epoch [{epoch}/{args.n_epochs}]",
        total=len(train_loader),
    ):

        X = batch[0]
        y = batch[1]
        X, y = X.cuda(), y.cuda()
        batchsize_cur = X.shape[0]

        model.train()
        model.zero_grad()

        enable_running_stats(model)
        out1, out2 = model(X)

        #####  task 1 #####
        loss1 = criterion(out1, y[:, 0])

        loss1.backward(retain_graph=True)
        # Task 1 Gradient Projections 
        kk = 0 
        grad1_projection = {}
        for k, (m,params) in enumerate(model.named_parameters()):
            # print(m)
            if len(params.size())==4:
                # print(m)
                sz =  params.grad.data.size(0)
                gp = torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                grad1_projection[m] = gp
                
                kk+=1
                params.grad.zero_()
                # print(grad1_projection.keys())
            # elif len(params.size())==1 and task_id !=0:
            #     params.grad.data.fill_(0)

        # print(grad1_projection.keys())     
        ##### task 2 #####
        loss2 = criterion(out2, y[:, 1])
        loss2.backward()

        #Task 2 Gradient Projections 
        kk = 0 
        grad2_projection = {}
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4:
                sz =  params.grad.data.size(0)
                gp = torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                grad2_projection[m] = gp
                
                kk+=1
        # print(grad2_projection) 

        all_losses_1 += loss1.detach().cpu().numpy() * batchsize_cur
        all_losses_2 += loss2.detach().cpu().numpy() * batchsize_cur

        #Gradient Deconfliction
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        final_grad = grad1_projection
        for name in final_grad:
            # print(name)
            # if cos(grad1_projection[name], grad2_projection[name]).sum() > 0:
            if torch.dot(torch.flatten(grad1_projection[name]),torch.flatten(grad2_projection[name])) > 0:
                # if torch.norm(grad1_projection[name]) < torch.norm(grad2_projection[name]):
                #     final_grad[name] = grad1_projection[name]
                # else:
                #     final_grad[name] = grad2_projection[name]
                final_grad[name] = grad2_projection[name]+grad1_projection[name] 
                
            else:
                final_grad[name].fill_(0)

        # Restore Positive Transfer Gradient
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4:
                # print(m)
                params.grad.data = final_grad[m]
            elif len(params.size())==1 and 'task' not in m:
            #     # print(m)
            # else:
            #   if 'task' not in name:
                params.grad.data.fill_(0)
      
        shared_optimizer.step()
        classifier_optimizer.step()
        model.zero_grad()


@torch.no_grad()
def test():

    model.eval()

    acc_1 = 0
    acc_2 = 0

    with torch.no_grad():

        for (it, batch) in enumerate(test_loader):
            X = batch[0]
            y = batch[1]
            X = X.cuda()
            y = y.cuda()

            out1_prob, out2_prob = model(X)
            out1_prob = F.softmax(out1_prob, dim=1)
            out2_prob = F.softmax(out2_prob, dim=1)
            out1 = out1_prob.max(1)[1]
            out2 = out2_prob.max(1)[1]
            acc_1 += (out1 == y[:, 0]).sum()
            acc_2 += (out2 == y[:, 1]).sum()

        acc_1 = acc_1.item() / len(test_loader.dataset)
        acc_2 = acc_2.item() / len(test_loader.dataset)

    return acc_1, acc_2


best_acc1 = 0
best_acc2 = 0
best_acc = 0

def get_representation_matrix_ResNet18 (net, device, x, y=None):
    net = net.encoder
    # Collect activations by forward pass
    x = x.to(device)
    net.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:1000] # ns=1000 examples 
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    # batch_list  = [100,100,100,100,100,100,100,100,500,500,500,1000,1000,1000,1000,1000,1000] #scaled
    # batch_list  = [50,50,50,50,50,50,50,50,250,250,250,500,500,500,500,500,500] #scaled
    batch_list  = [20,20,20,20,20,20,20,20,200,200,200,300,300,300,300,300,300] 
    # network arch 
    stride_list = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [36, 9,9,9,9, 9,5,5,5, 5,3,3,3, 3,2,2,2] 
    in_channel  = [ 1, 64,64,64,64, 64,128,128,128, 128,256,256,256, 256,512,512,512] 

    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 7
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)
    return mat_final 

def update_GPM (model, mat_list, threshold, feature_list):
    print ('Threshold: ', threshold) 
    # After First Task 
    for i in range(len(mat_list)):
        activation = mat_list[i]
        U,S,Vh = np.linalg.svd(activation, full_matrices=False)
        # criteria (Eq-5)
        sval_total = (S**2).sum()
        sval_ratio = (S**2)/sval_total
        r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
        feature_list.append(U[:,0:r])
    return feature_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_mat = []
feature_list =[]
threshold = np.array([0.965] * 20)
# Memory Update  
mat_list = get_representation_matrix_ResNet18(model, device, trainX, trainLabel)
feature_list = update_GPM(model, mat_list, threshold, feature_list)
for i in range(len(feature_list)):
    Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
    print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
    feature_mat.append(Uf)
for i in range(args.n_epochs):
    logging.info(f"Epoch [{i}/{args.n_epochs}]")
    losses = train(i, feature_mat)
    acc_1, acc_2 = test()
    logging.info(f"Accuracy task 1: {acc_1}, Accuracy task 2: {acc_2}")
    acc = (acc_1 + acc_2) / 2
    if acc > best_acc:
        log_str = f"Score improved from {best_acc} to {acc}. Saving model to {args.output_dir}"
        logging.info(log_str)
        # torch.save(model.state_dict(), f"{args.output_dir}/model.pt")
        best_acc = acc