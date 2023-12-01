# coding:utf8
from config import opt
import os
import torch as t
from data.dataset import ParticleTrajectory
from data import xlsx_operation
from models.lstmnet import LSTMNet
from models.lstmAttNet import LSTMATTNet
from models.coherence_loss import CoherenceLoss
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
import ipdb
import random
import numpy as np
import time
from torch import nn
import pandas as pd
from utils import metrics


# reset model weight
def weight_reset(self):
    if isinstance(self, nn.LSTM) or isinstance(self, nn.Linear):
        self.reset_parameters()


@t.no_grad() 
def test(**kwargs):

    import ipdb
    ipdb.set_trace()
    opt._parse(kwargs)

    # configure model
    in_dim = opt.in_dim
    hidden_dim = opt.hidden_dim
    n_layer = opt.n_layer
    n_class = opt.n_class
    device = opt.device

    if opt.use_attention:
        attention_win = opt.attention_win
        model = LSTMATTNet(in_dim, hidden_dim, n_layer, n_class, device, attention_win)
    else:
        model = LSTMNet(in_dim, hidden_dim, n_layer, n_class, device)
   
    print(model)

    # load model
    if opt.load_model_path:
        
        checkpoint = t.load(opt.load_model_path)
        best_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])

        print("==> Loaded checkpoint '{}'.".format(opt.load_model_path))
        print("==> Epoch: {}.".format(best_epoch+1))
        print("==> Best Accuracy:  {:.2f} %.".format(best_acc * 100))


    # accelerate model
    if opt.use_gpu:
        model.to(opt.device)

    # load test data,from feature extraction folder
    test_data = ParticleTrajectory(opt.test_data_root,per=np.array([0]),test=True)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    
    # get xlsx file name, for write the predict result in initial folder
    result = [os.path.join(opt.test_result_root, file) for file in os.listdir(opt.test_result_root)]
    result = sorted(result, key=lambda x: int(x.split('/')[-1].split('.')[-2])) 

    # test
    iou_threshold = opt.iou_threshold

    label_num = t.zeros([len(test_data),1])    
    pred_num = t.zeros([len(test_data),1])    
    tracklet_num = t.zeros([len(test_data),1])    
    trackletPred_num = t.zeros([len(test_data),1])    
    precision_class = t.zeros([len(iou_threshold),len(test_data),n_class])

    for ii,(data,label) in enumerate(test_dataloader):
        if opt.use_gpu:
            input = data.to(opt.device)
            label = label.to(opt.device)
        else: 
            input = data
            label = label

	# input:float32
        input = input.float()
        # label:long
        label = label.view(-1).long()

        score, lstm_out  = model(input)
        _, pred = t.max(score, 1)
        
        # frameAcc
        label_num[ii] = len(label)
        pred_num[ii] = (pred == label).sum()
        # trackletAcc
        tracklet_num[ii],trackletPred_num[ii] = metrics.getTrackletPredNum(label,pred)
        # precision
        for jj in range(len(iou_threshold)):
            precision_class[jj,ii,:] = metrics.getPrecision(label,pred,n_class,iou_threshold[jj])



        # restore label start from 1
        pred_arr = pred.cpu().numpy() + 1

        # record the result
        # simulation data,write predict label in column 4(4+1),have groundtruth in column 3(3+1)
        # real data,write predict label in column 4(4+1),have groundtruth in column 3(3+1),if has no groundtruth,label=-1
        xlsx_operation.write_excel_xlsx_col(result[ii],'Sheet1',pred_arr,1,4)

    frameAcc = metrics.getFrameAcc(label_num,pred_num)
    trackletAcc = metrics.getTrackletAcc(tracklet_num,trackletPred_num)
    mAP = t.zeros([1,len(iou_threshold)])
    for jj in range(len(iou_threshold)):
       mAP[0,jj] = metrics.getmAP(precision_class[jj])
    print("==> test frameAcc:  {:.2f} %.".format(frameAcc * 100))
    print("==> test trackletAcc:  {:.2f} %.".format(trackletAcc * 100))
    print("==> iou threshold:  {}.".format(iou_threshold))
    np.set_printoptions(precision=2, suppress=True)
    print("==> test mAP:  {} %.".format(mAP.numpy() * 100))

def train(**kwargs):
    # get consistent result
    random.seed(1)
    np.random.seed(1)
    t.manual_seed(1)
    t.cuda.manual_seed(1)
    t.cuda.manual_seed_all(1)

    opt._parse(kwargs)

#    import ipdb
#    ipdb.set_trace()

    # start visdom
    #vis = Visualizer(opt.env,port = opt.vis_port)
    
    
    # compute the train time
    t.cuda.synchronize()
    start = time.time()

    # repeat the training process for ksplit times
    ksplit = opt.ksplit
    train_loss_Array=[]
    train_acc_Array=[]
    val_loss_Array=[]
    val_acc_Array=[]
    best_acc_Array=[]
    for kk in range(ksplit):

        print('ksplit={}'.format(kk+1)) 

        # setp3: load data
        data = [os.path.join(opt.train_data_root, sample) for sample in os.listdir(opt.train_data_root)]
        # shuffle the data, and split into train set and validation set.
        per = np.random.permutation(len(data))
        print(per[0:10])
        trainset_ratio = opt.trainset_ratio
        train_data = ParticleTrajectory(opt.train_data_root,per,train=True,ratio=trainset_ratio,k=kk)
        val_data = ParticleTrajectory(opt.train_data_root,per,train=False,ratio=trainset_ratio,k=kk)
        train_dataloader = DataLoader(train_data,opt.batch_size,
                            shuffle=True,num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_data,opt.batch_size,
                            shuffle=False,num_workers=opt.num_workers)        
        
        for attention_win in opt.attention_win:
            # step1: configure model    
            in_dim = opt.in_dim
            hidden_dim = opt.hidden_dim
            n_layer = opt.n_layer
            n_class = opt.n_class
            device = opt.device

            if opt.use_attention:
                #attention_win = opt.attention_win
                print("attention_win:{}".format(attention_win))
                model = LSTMATTNet(in_dim, hidden_dim, n_layer, n_class, device, attention_win)
            else:
                model = LSTMNet(in_dim, hidden_dim, n_layer, n_class, device)
            
            print(model)

            if opt.load_model_path:
                
                checkpoint = t.load(opt.load_model_path)
                best_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                model.load_state_dict(checkpoint['model_state_dict'])

                print("==> Loaded checkpoint '{}'.".format(opt.load_model_path))
                print("==> Epoch: {}.".format(best_epoch+1))
                print("==> Best Accuracy:  {:.2f} %.".format(best_acc * 100))

        
            if opt.use_gpu:
                model.to(opt.device)

        
            # step2: criterion and optimizer
            criterion = t.nn.CrossEntropyLoss()
            lr = opt.lr
            optimizer = model.get_optimizer(lr, opt.weight_decay)
            # dynamically adjust the learning rate
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=1,patience=5,factor=0.5,eps=1e-06)

                



            # # setp3: load data
            # data = [os.path.join(opt.train_data_root, sample) for sample in os.listdir(opt.train_data_root)]
            # # shuffle the data, and split into train set and validation set.
            # per = np.random.permutation(len(data))
            # print(per[0:10])
            # trainset_ratio = opt.trainset_ratio
            # train_data = ParticleTrajectory(opt.train_data_root,per,train=True,ratio=trainset_ratio,k=kk)
            # val_data = ParticleTrajectory(opt.train_data_root,per,train=False,ratio=trainset_ratio,k=kk)
            # train_dataloader = DataLoader(train_data,opt.batch_size,
                                # shuffle=True,num_workers=opt.num_workers)
            # val_dataloader = DataLoader(val_data,opt.batch_size,
                                # shuffle=False,num_workers=opt.num_workers)
            
            

            # reset parameter
            all_train_loss = []
            all_train_acc = []
            all_val_loss = []
            all_val_acc = []
            best_acc = 0.0
            

            for epoch in range(opt.max_epoch):
               
                # train
                model.train() 
                running_loss = 0
                running_acc = 0
                num_label_train = 0
                for ii,(data,label) in enumerate(train_dataloader):

                    if opt.use_gpu:
                        input = data.to(opt.device)
                        target = label.to(opt.device)
                    else:
                        input = data
                        target = label
                
                    # input:float32,format:(1,56,2)/(256,20,3) batchsize,seq,feature_dim
                    input = input.float()

                    # target:long,format:(1,56,1)-->(56) / (256,20,1)-->(5120)
                    raw_target = target
                    target = target.view(-1).long()

                    #score = model(input)
                    #loss = criterion(score,target)


                    score, lstm_out  = model(input)
                    loss = criterion(score,target)
                    
                    if opt.use_customloss:
                        delta = opt.delta
                        alpha = opt.alpha
                        beta = opt.beta
                        extra_criterion = CoherenceLoss(device)
                        coherence_loss = extra_criterion(lstm_out,raw_target,delta)
                        loss = alpha * loss + beta * coherence_loss
                
                    running_loss += loss.item() * target.size(0)
                    _, pred = t.max(score, 1)
                    num_correct = (pred == target).sum()
                    running_acc += num_correct.item()
                    num_label_train += len(target)

                    #print('train frame len:{}, total:{}'.format(len(target),num_label_train))
                    # vis.log('train frame len:{}, total:{}'.format(len(target),num_label_train))

                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                    if (ii + 1)%opt.print_freq == 0:
                        print('[{}/{}] time: {} Loss: {:.6f}, Acc: {:.6f}'.format(
                             epoch + 1, opt.max_epoch, ii+1 , running_loss / num_label_train, running_acc / num_label_train))
                    #    vis.log('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    #            epoch + 1, opt.max_epoch, running_loss / num_label_train, running_acc / num_label_train))

                        
                    # start debug mode
                    if os.path.exists(opt.debug_file):
                        import ipdb
                        ipdb.set_trace()

                print('loop ii: {}, finish {} epoch, train_Loss: {:.6f}, train_Acc: {:.6f}'.format(
                             ii+1, epoch + 1, running_loss / num_label_train, running_acc / num_label_train))
                #vis.log('finish {} epoch, train_Loss: {:.6f}, train_Acc: {:.6f}'.format(
                #            epoch + 1, running_loss / num_label_train, running_acc / num_label_train))

                train_loss = running_loss / num_label_train
                all_train_loss.append(train_loss)
                train_acc = running_acc / num_label_train
                all_train_acc.append(train_acc)
                #vis.plot('train_loss', 'epoch','loss',epoch,train_loss)
                #vis.plot('train_accuracy', 'epoch','accuracy',epoch,train_acc)

                # validate
                model.eval()
                eval_loss = 0
                eval_acc = 0
                num_label_val = 0

                for ii, (val_input, label) in enumerate(val_dataloader):
                    if opt.use_gpu:
                        with t.no_grad():
                            val_input = val_input.to(opt.device)
                            target = label.to(opt.device)
                    else:
                        with t.no_grad():
                            val_input = val_input
                            target = label

                    # val_input:float32,format:(1,56,2) batchsize,seq,feature_dim
                    val_input = val_input.float()

                    # target:long,format:(1,56,1)-->(56) / (256,20,1)-->(5120)
                    #target = target.squeeze(-1).squeeze(-2).long()
                    raw_target = target
                    target = target.view(-1).long()


                    score, lstm_out  = model(val_input)
                    loss = criterion(score,target)

                    if opt.use_customloss:
                        delta = opt.delta
                        alpha = opt.alpha
                        beta = opt.beta
                        extra_criterion = CoherenceLoss(device)
                        coherence_loss = extra_criterion(lstm_out,raw_target,delta)
                        loss = alpha * loss + beta * coherence_loss


                    eval_loss += loss.item() * target.size(0)
                    _, pred = t.max(score, 1)
                    num_correct = (pred == target).sum()
                    eval_acc += num_correct.item()
                    num_label_val += len(target)
                    #print('val frame len:{}, total:{}'.format(len(target),num_label_val))
                    #vis.log('val frame len:{}, total:{}'.format(len(target),num_label_val))

                val_loss = eval_loss / num_label_val
                all_val_loss.append(val_loss)
                val_acc = eval_acc / num_label_val
                all_val_acc.append(val_acc)
                #vis.plot('val_loss','epoch','loss',epoch,val_loss)
                #vis.plot('val_accuracy','epoch','accuracy',epoch,val_acc)

                # update lr if val_loss increase beyond patience
                scheduler.step(val_loss) 
                lr = optimizer.param_groups[0]['lr']

                print('finish {} epoch, lr: {}, val_Loss: {:.6f}, val_Acc: {:.6f}'.format(
                            epoch + 1, lr, val_loss, val_acc))
                #vis.log('finish {} epoch, lr: {}, val_Loss: {:.6f}, val_Acc: {:.6f}'.format(
                #            epoch + 1, lr, val_loss, val_acc))



                # save model base on val_acc
                if best_acc < val_acc:
                    best_acc = val_acc
                    t.save({ 'epoch': epoch,
                             'model_state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'best_acc': best_acc},
                             os.path.join('./checkpoints/', 'attwin['+str(attention_win)+']'+opt.train_data_root.split('/')[-2]+'_'+str(kk)+'_best_checkpoints.pth'))



            print('ksplit:{},Best val Acc: {:4f}'.format(kk+1,best_acc))
            best_acc_Array.append(best_acc)
        
            print('train_loss:{},train_acc:{},val_loss:{},val_acc:{}'.format(all_train_loss,all_train_acc,all_val_loss,all_val_acc))
            train_loss_Array.append(all_train_loss)
            train_acc_Array.append(all_train_acc)
            val_loss_Array.append(all_val_loss)
            val_acc_Array.append(all_val_acc)
        
    
    print('train_loss:{}'.format(train_loss_Array))
    print('train_acc:{}'.format(train_acc_Array))
    print('val_loss:{}'.format(val_loss_Array))
    print('val_acc:{}'.format(val_acc_Array))

    allResult=[]
    allResult.extend(train_loss_Array)
    allResult.extend(train_acc_Array)
    allResult.extend(val_loss_Array)
    allResult.extend(val_acc_Array)
    df = pd.DataFrame(allResult)
    xlsxName = 'attwin['+str(attention_win)+']'+opt.train_data_root.split('/')[-2] + '_'+'allResult.xlsx'
    df.to_excel(xlsxName, index=False)

    #print('best val acc===>')
    #print(best_acc_Array)
    print('attention win:{}, best val acc:{}'.format(attention_win,best_acc_Array))
    
    # compute train time
    t.cuda.synchronize()
    time_elapsed = time.time()-start
    #print(time_elapsed)
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def help():
     """
     python file.py help
     """
    
     print("""
     usage : python file.py <function> [--args=value]
     <function> := train | test | help
     example: 
             python {0} train --vis_port=6666
             python {0} test --load_model_path='./checkpoints/xxx.pth'
             python {0} help
     avaiable args:""".format(__file__))

     from inspect import getsource
     source = (getsource(opt.__class__))
     print(source)

if __name__=='__main__':

    import fire

    fire.Fire()

