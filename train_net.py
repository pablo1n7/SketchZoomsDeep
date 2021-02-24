from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from utils.SZDataset import *

from utils.TripletLoss import *
from utils.ContrastiveLoss import *

from utils.utilities import VisdomLinePlotter

from net.SketchNetworkResnet import *
from net.SketchNetwork import *
from net.SketchNetworkVGG import *


import click
import sh

from torch import optim
import numpy as np
import glob
import torch
import os


class Config():
    train_batch_size = 64
    val_batch_size   = 64
    train_data_n     = 14400 #576 #545300
    val_data_n       = 1440  #144  #68099
    train_number_epochs = 1000
    
    

@click.command()
@click.option('--netname', default="alexnet", prompt="Network", help='alexnet, vgg, resnet')
@click.option('--nepoch', default=100, help='30, 40, 50, 60')
@click.option('--lossname', default="triplet", prompt="Loss", help='triplet, contrast, crossEntropyLoss')
@click.option('--opt',default="adm", prompt="op", help='adam, sgd')
@click.option('--lr',default=0.0001, prompt="lr", help='0.00001, 0.0001, 0.001')
@click.option('--dirCheckpoint',default="checkpoint_network", help='')
@click.option('--dirDataset',default="data/", help='')
@click.option('--device',default="cuda", help='cuda, cuda:0, cuda:1, cpu')
@click.option('--envplotter',default="main", help='cualquiernombre')

#python train_net.py --netname alexnet --nepoch 1000 --lossname triplet --opt adm --lr 0.0001 --device cuda --envplotter main

#python train_net.py --netname resnet --nepoch 1000 --lossname triplet --opt adm --lr 0.0001 --device cuda --envplotter resnet

#python train_net.py --netname alexnet --nepoch 1000 --lossname crossEntropyLoss --opt adm --lr 0.0001 --device cuda --envplotter main

#python train_net.py --netname alexnet --nepoch 1000 --lossname contrast --opt adm --lr 0.0001 --device cuda --envplotter main



    
def main(netname, nepoch, lossname, opt, lr, dircheckpoint, dirdataset, device, envplotter):
    print(20*"-")
    print("netname:", netname)
    print('n_epoch:', nepoch)
    print("loss:", lossname)
    print("opt:", opt)
    print("lr:", lr)
    print("dircheckpoint:", dircheckpoint)
    print("dirDataset:", dirdataset)
    print("device:", device)
    print('envplotter:', envplotter)
    print(20*"-")

    Config.train_number_epochs = nepoch
    
    networks = {"alexnet":SketchNetwork, 
                "resnet":SketchNetworkResnet,
                "vgg": SketchNetworkVGG}
    
    net = networks[netname]()
    #net = nn.DataParallel(net)
    net = net.to(device)
    
    image_size = 224
    if netname == "inception":
        image_size = 299
        print(image_size)
    
    criterion_triplet = nn.TripletMarginLoss(margin=1.0)
    criterion_contrast = ContrastiveLoss()
    criterion_CE = nn.CrossEntropyLoss()
    
    losses ={"triplet": lambda x,y,z: criterion_triplet(x, y, z), 
             "contrast": lambda x,y,z: (
                 criterion_contrast(x, y, torch.ones(Config.train_batch_size).to(device)) + 
                 criterion_contrast(x, z, -1*torch.ones(Config.train_batch_size).to(device))),
            "crossEntropyLoss": lambda out0, out1:(
                 criterion_CE(torch.cat((out0, out1)), 
                              torch.cat((torch.zeros(Config.train_batch_size),
                                            torch.ones(Config.train_batch_size))).to(device).long())
                              )}
    
    criterion = losses[lossname]
    
    uses_triple= False
    if lossname=="triplet":
        uses_triple= True
    
    
    if opt!='adm':
        optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
    else:
        optimizer = optim.Adam(net.parameters(), lr = lr)
    
    plotter = VisdomLinePlotter(env_name=envplotter, port=8097)
    
    epoch_loss = 0
    valid_epoch_loss = 0
    best_loss = -1
    num_batch = 1
    num_batch_val = 1

    sh.mkdir("-p", dircheckpoint)
    files_checkpoints = np.array(sorted(glob.glob(dircheckpoint+"/*{}_{}*".format(netname, lossname))))
    if (files_checkpoints.shape[0]):
        checkpoint = torch.load(files_checkpoints[-1])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(files_checkpoints[-1])
        
    
    transf = transforms.Compose([transforms.RandomRotation((-45,45), fill=(255, 255, 255, 1)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.Resize((image_size, image_size)), 
                                 transforms.ToTensor()])
    
    
    
    #DATA LOADERS
    sketch_dataset = SketchZoomDataset(data_sketch_root= "data/",
                                            net=net, 
                                            plotter=None, 
                                            n=Config.train_data_n, 
                                            stage= "train", 
                                            image_size=image_size, 
                                            triplet=uses_triple, 
                                            categories=["Airplane","Bag","Cap","Car","Chair","Earphone","Guitar","Knife","Lamp","Laptop","Motorbike","Mug","Pistol","Rocket","Skateboard","Table"], 
                                            transform= transf, 
                                            device=device)

    
    train_dataloader = DataLoader(sketch_dataset,
                            shuffle= True,
                            num_workers= 0,
                            batch_size= Config.train_batch_size,
                            drop_last=True)
    
    
    
    sketch_dataset_test = SketchZoomDataset(data_sketch_root= "data/",
                                            net=net, 
                                            plotter=None, #plotter, 
                                            n=Config.val_data_n, 
                                            stage= "test", 
                                            image_size=image_size, 
                                            triplet=uses_triple, 
                                            categories=["Airplane","Bag","Cap","Car","Chair","Earphone","Guitar","Knife","Lamp","Laptop","Motorbike","Mug","Pistol","Rocket","Skateboard","Table"], 
                                            transform= transf, 
                                            device=device)
    
    validation_dataloader = DataLoader(sketch_dataset_test,
                            shuffle=True,
                            num_workers=0,
                            batch_size=Config.val_batch_size,
                            drop_last=True)
    
    
    for epoch in range(1, Config.train_number_epochs):
        print('EPOCH', epoch)
        valid_epoch_loss=0
        epoch_loss = 0
        net.train()

        #TRAIN
        for i, data in enumerate(train_dataloader):
            sketch_dataset.net = net
            net.train()
            
            img0, img1, img2 = data
            img0, img1 , img2 = Variable(img0).to(device), Variable(img1).to(device) , Variable(img2).to(device)
            optimizer.zero_grad()

            if lossname != 'crossEntropyLoss':
                output1, output2, output3 = net(img0, img1, img2, img1.size()[0])
                loss = criterion(output1, output2, output3)
            else:
                output1, output2, res0 = net.forward_two_binary(img0, img1, img1.size()[0])
                output1, output3, res1 = net.forward_two_binary(img1, img2, img2.size()[0])
                loss = criterion(res0, res1)
                
            
            distances_negativa = F.pairwise_distance(output1, output3)
            distances_negativa = distances_negativa.data.cpu().numpy().flatten()
            distances_positiva = F.pairwise_distance(output1, output2)
            distances_positiva = distances_positiva.data.cpu().numpy().flatten()
            
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            
            print('*' * 20)
            print(" nro Batch {} --  Current loss {}\n".format(i,loss.item()))
            num_batch = num_batch +1
            plotter.plot('Distance mean', str(0), num_batch, np.mean(distances_positiva) ,"Batchs")
            plotter.plot('Distance mean', str(1), num_batch, np.mean(distances_negativa) ,"Batchs")
            plotter.plot('Batchs loss', str(epoch), i + 1, loss.item(),"Batchs")
            if i !=0 and i%9 == 0:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "{}/{}_{}_{}.pkl".format(dircheckpoint, "current_batch_checkpoints", netname, lossname))
                print("save net, checkpoint Batch")

        del i, data, distances_negativa, distances_positiva, output1, output2, output3, img0, img1, img2
        
        net.eval() 
        with torch.no_grad():
            
            sketch_dataset.net = net

            for i, data in enumerate(validation_dataloader):
                
                img0, img1 , img2 = data
                img0, img1 , img2 = Variable(img0).to(device), Variable(img1).to(device) , Variable(img2).to(device)
                
                if lossname != 'crossEntropyLoss':
                    output1, output2, output3 = net(img0, img1, img2, img1.size()[0])
                    loss = criterion(output1, output2, output3)
                else:
                    output1, output2, res0 = net.forward_two_binary(img0, img1, img1.size()[0])
                    output1, output3, res1 = net.forward_two_binary(img0, img2, img2.size()[0])
                    loss = criterion(res0, res1)
                    
                   
                distances_negativa = F.pairwise_distance(output1, output3)
                distances_negativa = distances_negativa.data.cpu().numpy().flatten()
                distances_positiva = F.pairwise_distance(output1, output2)
                distances_positiva = distances_positiva.data.cpu().numpy().flatten()
                num_batch_val = num_batch_val +1
                
                plotter.plot('Distance mean Valid', 
                             str(0), 
                             num_batch_val, 
                             np.mean(distances_positiva[0]), 
                             "Batchs")
                
                plotter.plot('Distance mean Valid', 
                             str(1), 
                             num_batch_val, 
                             np.mean(distances_negativa[0]),
                             "Batchs")
                print(" nro Valid Batch{} --  Valid loss {}\n".format(i+1,loss.item()))
                valid_epoch_loss+= loss.item()

            del i, data, distances_negativa, distances_positiva, output1, output2, output3, img0, img1, img2
            

        #END TRAIN    
        current_epoch_loss = epoch_loss / (Config.train_data_n // Config.train_batch_size)
        current_epoch_loss_val = valid_epoch_loss / (Config.val_data_n // Config.val_batch_size)

        print("Epoch number {}\n Current loss average {}\n".format(epoch, current_epoch_loss))
        print("Epoch number {}\n Current loss val average {}\n".format(epoch, current_epoch_loss_val))
        plotter.plot('Epochs loss ', 'train epoch', epoch, current_epoch_loss, "Epochs")
        plotter.plot('Epochs loss ', 'valid epoch', epoch, current_epoch_loss_val, "Epochs")
        
        #SAVE NET WITH BEST LOSS
        if(best_loss == -1 or current_epoch_loss < best_loss):
            torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "{}/{}_{}_{}.pkl".format(dircheckpoint, "checkpoints", netname, lossname))
        
            best_loss = current_epoch_loss
            print("save net, loss: {}".format(best_loss))    

if __name__ == '__main__':
    main()
