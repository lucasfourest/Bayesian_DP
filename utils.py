import numpy as np
from tqdm import tqdm 
import itertools

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import required
from torch.autograd import Variable
from torch.autograd import Function

from accountant import *



def train(args,trainloader, testloader, net, powers=None):

    n_epochs, lr, batch_size = args.ne, args.lr, float(args.bs)
    total_steps=n_epochs * len(trainloader)
    q = batch_size / len(trainloader.dataset)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=lr)
    use_cuda=torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    
    accuracies = []
    if powers is not None: # accountant enabled
        privacy_costs=np.zeros(powers.shape[0])
        target_delta, target_eps = 1e-5,1
    eps_history=[]
    delta_history=[]
    max_grad_norm = args.C
    
    for epoch in tqdm(range(n_epochs)):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            inputv = Variable(inputs)
            labelv = Variable(labels.long().view(-1) % 10)
              

            optimizer.zero_grad()
            outputs = net(inputv)
            loss = criterion(outputs, labelv)
            

            if powers is not None:
                grads_est = []
                num_subbatch = args.n_samples
                for j in range(num_subbatch):
                    grad_sample = torch.autograd.grad(
                        loss[np.delete(range(int(batch_size)), j)].mean(), 
                        [p for p in net.parameters() if p.requires_grad], 
                        retain_graph=True
                    )
                    with torch.no_grad():
                        grad_sample = torch.cat([g.view(-1) for g in grad_sample])
                        grad_sample /= max(1.0, grad_sample.norm().item() / max_grad_norm)
                        grads_est += [grad_sample]
                


            (loss.mean()).backward()
            running_loss += loss.mean().item()


            if powers is not None:
                with torch.no_grad():
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                p.grad += torch.randn_like(p.grad) * (args.sigma* max_grad_norm)
                
            optimizer.step()

            if powers is not None:
                with torch.no_grad():
                    batch_size = float(len(inputs))
                    pairs = list(zip(*itertools.combinations(grads_est, 2)))
                    ldistr=(torch.stack(pairs[0]), args.sigma*max_grad_norm)
                    rdistr=(torch.stack(pairs[1]), args.sigma*max_grad_norm)
                    
                    
                    privacy_costs += np.array([get_cost(power,ldistr,rdistr,total_steps,q\
                                                               ) for power in powers])
                    eps=get_epsilon(powers,privacy_costs, target_delta=target_delta)
                    delta=get_delta(powers,privacy_costs, target_eps=target_eps)
                    eps_history.append(eps), delta_history.append(delta)
            
        if epoch % args.eval ==0:    
            running_eps = eps_history[-1] if powers is not None else None
            running_delta = delta_history[-1] if powers is  not None else None
            print("Epoch: %d/%d. Loss: %.3f. Privacy (𝜀,𝛿): %s %s" %
                (epoch + 1, n_epochs, running_loss / len(trainloader), running_eps, running_delta))
                    
            acc = test(testloader, net)
            accuracies += [acc]
            print("Test accuracy is %d %%" % acc)

    print('Finished Training')
    return net.cpu(), accuracies, eps_history, delta_history


def test(testloader, net):

    correct = 0.0
    total = 0.0
    for data in testloader:
        img, labels = data
        if torch.cuda.is_available():
            img = img.cuda()
            labels = labels.cuda()         
        # outputs = net(Variable(img))
        output = net(img)
        _, predicted = torch.max(output,dim=-1)
        #print(predicted.cpu().numpy())
        #print(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == (labels.long().view(-1) % 10)).sum()
        #print torch.cat([predicted.view(-1, 1), (labels.long() % 10)], dim=1)

    print('Accuracy of the network on test images: %f %%' % (100 * float(correct) / total))
    return 100 * float(correct) / total