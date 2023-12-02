import numpy as np
import os
import sys
import gc

from tqdm import tqdm
import shutil
import librosa as lb

import torch

from torch.backends import cudnn
from torch.autograd import Variable
from graphs.models.torchvggish import VGGish
from graphs.models.crnn import CRNN

import apex
from apex import amp, optimizers
from torch.nn.parallel import DistributedDataParallel as THDDP
from apex.parallel import DistributedDataParallel as APDDP

from datasets.fma import FmaDB
from datasets.prefetcher import data_prefetcher

from torch.optim import lr_scheduler

from utils.metrics import AverageMeter, IOUMetric
from utils.misc import print_cuda_statistics
from utils.profile_wrapper import cprofile_wrapper
from utils.transform import gl_cqt

from agents.base import BaseAgent
from graphs.optimizer import OpenAIAdam, RAdam
from graphs.losses import LabelSmoothingLoss
from datasets.norm import norm
import soundfile as sf

from torch.utils.tensorboard import SummaryWriter



class FmaAgent(BaseAgent):
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        super().__init__(config)
       

        # Create an instance from the Model

        # define model
        self.sep_only = config.sep_only
        self.pitch_only = config.pitch_only
        assert not (self.sep_only and self.pitch_only)
        self.half = config.half        
        distributed = config.distributed
        self.checkpoint_dir = config.checkpoint_dir
        self.load_from = config.load_from
        print("checkpoint dir: ", self.checkpoint_dir)
        model = globals()[config.model.name]()
        model.load_state_dict(torch.load("pretrained_weights/vggish.pth"), strict=False)
         

        if config.opt == "radam":
            optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.opt == "adam":
            optimizer = OpenAIAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, warmup=0.99, t_total=160)
        if config.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
        local_rank = 0
        if distributed:
            local_rank = int(sys.argv[1].split('=')[-1])
            map_location = {f"cuda:0":f"cuda:{local_rank}"}
            torch.cuda.set_device(local_rank)

            ### CAUTHION WORLD_SIZE
            world_size = int(os.environ['WORLD_SIZE'])
            self.world_size = world_size
            torch.distributed.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
            model = apex.parallel.convert_syncbn_model(model)
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
            model = model.cuda()
            if self.half:
                optimizer = OpenAIAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            model = APDDP(model, delay_allreduce=True)
            if True:
                self.criterion = torch.nn.CrossEntropyLoss().to(local_rank)

        self.local_rank = local_rank
        self.model = model
        self.optimizer = optimizer

        # Create an instance from the data loader
        
        if 'TESTING_FLAG' in os.environ and int(os.environ['TESTING_FLAG']) == 1:
            pass
        else:
            self.data_source = FmaDB(self.config.sep_data)
            print('###DATASOURCE LOADED####')

        self.current_epoch = 0
        self.current_iteration = 0

        if hasattr(config, "load_from") and config.load_from is not None:
            print(f'##### LOADING CHECKING POINT {config.load_from}')
            if 'map_location' in locals():
                self.load_checkpoint(filename=config.load_from, map_location=map_location)
            else:
                self.load_checkpoint(filename=config.load_from)

        ### TFBoard
        if not distributed or local_rank == 0:
            self.log_writer = SummaryWriter(os.path.join(self.checkpoint_dir, "tblogs"), flush_secs=30)

    def save_checkpoint(self, filename='checkpoint_{}.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        checkpoint_dir = self.checkpoint_dir
        torch.save(state, os.path.join(checkpoint_dir, filename.format(str(self.current_epoch + 1))))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename, map_location=lambda d:d):
        filename = os.path.join(self.config.checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            ###DEBUG
            #self.optimizer.load_state_dict(checkpoint['optimizer'])

        except OSError as e:
            
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self, mode):
        """
        This function will the operator
        :return:
        """
        assert mode in ['train', 'test', 'random']
        try:
            if mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        self.model.train()
        if self.load_from is not None:
            pass
            # test_metric = self.evaluate()
            # print(test_metric)

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            loss_metric = self.train_one_epoch(sep_only=self.sep_only, pitch_only=self.pitch_only)
            cmd = r"""echo "{}" >> {}""".format(repr(loss_metric)+f"epoch{epoch}",os.path.join(self.checkpoint_dir, "log"))
            '''
            test_metric = self.evaluate()
            cmd = r"""echo "{}" >> {}""".format(repr(test_metric),os.path.join(self.checkpoint_dir, "log"))
            os.system(cmd)
            '''
            
            if self.local_rank == 0:
                self.save_checkpoint()
                os.system(cmd)
                test_metric = self.evaluate()
                cmd = r"""echo "{}" >> {}""".format(repr(test_metric)+f"[test]{epoch}[test]",os.path.join(self.checkpoint_dir, "log"))
                os.system(cmd)

    def evaluate(self, sep_only=False):
        self.model.eval()
        data_src = self.data_source
        data_loader = data_prefetcher(data_src.test_loader)
        iterations = data_src.test_iterations
        local_rank = self.local_rank

        epoch_acc = AverageMeter()


        for idx, batch in enumerate(tqdm(data_loader, total=iterations, dynamic_ncols=False, ncols=36)):

            input_ = batch['feat']
            input_ = input_.half() if self.half else input_.float()
            input_ = input_.to(self.local_rank)

            with torch.no_grad():
                logits = self.model(input_) 
            label = batch['label'].long()

            logits_np = logits.detach().cpu().numpy()
            label_np = label.numpy() 
            pred_np = np.argmax(logits_np, axis=-1) 
            acc = torch.Tensor([np.mean(pred_np == label_np)]).to(self.local_rank)
            torch.distributed.barrier()
            torch.distributed.reduce(acc, dst=0)
            torch.distributed.barrier()
            acc = acc / self.world_size
            epoch_acc.update(float(acc.cpu()))
            self._set_trace()
        return epoch_acc.avg
    
    def _set_trace(self):
        if self.local_rank == 0:
            if open("/tmp/debug").read().startswith("nmsl"):
                from remote_pdb import RemotePdb
                RemotePdb('127.0.0.1', 4444+self.local_rank).set_trace()
     


        
    @cprofile_wrapper("/root/train_one_epoch.prof")
    def train_one_epoch(self, sep_only=False, pitch_only=False):
        """
        One epoch training function
        """
        ## prepare ###
        self.model.train()
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        data_src = self.data_source
        data_loader = data_prefetcher(data_src.loader)
        iterations = data_src.iterations
        local_rank = self.local_rank
        
        torch.backends.cudnn.benchmark = True
        for idx, batch in enumerate(tqdm(data_loader, total=iterations, ncols=36)):
            profile = False
            if profile == True and idx > 128:
                return epoch_loss.avg, epoch_acc.avg
            input_ = batch['feat']
            input_ = input_.half() if self.half else input_.float()
    
            input_ = input_.to(self.local_rank)

            logits = self.model(input_) 
        
            label = batch['label'].long().to(local_rank) 
            loss_scale = 1.

            if type(self.criterion) == torch.nn.CrossEntropyLoss: 
                cnt_loss = self.criterion(logits, label)


            # optimizer
            self.optimizer.zero_grad()
            if self.half:
                with amp.scale_loss(cnt_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                cnt_loss.backward()
            self.optimizer.step()
            
            label_np = label.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            pred_np = np.argmax(logits_np, axis=-1)
            acc = torch.Tensor([np.mean(pred_np == label_np)]).to(self.local_rank)
                
            torch.distributed.barrier()
            cnt_loss_detach = cnt_loss.detach()
            torch.distributed.reduce(cnt_loss_detach, dst=0) 
            torch.distributed.barrier()
            cnt_loss_detach = cnt_loss_detach / self.world_size
            epoch_loss.update(cnt_loss_detach.item())

            torch.distributed.barrier()
            torch.distributed.reduce(acc, dst=0)
            torch.distributed.barrier()
            acc = acc / self.world_size
            epoch_acc.update(float(acc.cpu()))

            self.current_iteration += 1 

            if local_rank == 0:
                if open("/tmp/debug").read().startswith("nmsl"):
                    from remote_pdb import RemotePdb
                    RemotePdb('127.0.0.1', 4444+local_rank).set_trace()
                    from utils.plot import plotCurveMat
                    plotCurveMat([pred_np.reshape(-1), label_np.reshape(-1)], labels=['pred', 'label'])


        torch.cuda.empty_cache()
        return epoch_loss.avg, epoch_acc.avg
    def _norm_wav(self, arr):
        mean = float(arr.max().detach().cpu()) # detach for disable gradient through max op
        return arr / mean
    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        metrics = IOUMetric(self.config.num_classes)

        for x, y in tqdm_batch:
            if self.cuda:
                pass
                #x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
            x, y = Variable(x), Variable(y)
            # model
            pred = self.model(x)
            # loss
            cur_loss = self.loss(pred, y)

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            _, pred_max = torch.max(pred, 1)
            metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

            epoch_loss.update(cur_loss.item())

        epoch_acc, _, epoch_iou_class, epoch_mean_iou, _ = metrics.evaluate()
        self.summary_writer.add_scalar("epoch_validation/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_validation/mean_iou", epoch_mean_iou, self.current_iteration)

        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + " - acc-: " + str(
            epoch_acc) + "- mean_iou: " + str(epoch_mean_iou) + "\n iou per class: \n" + str(
            epoch_iou_class))

        tqdm_batch.close()

        return epoch_mean_iou, epoch_loss.val


    def test_one_file(self, filepair):
        self.model.eval()

        y, sr = lb.core.load(filepair[0], sr=44100, mono=False)     
        if len(y.shape) == 1:
            input_np = np.repeat(y[::,np.newaxis], 2, axis=-1)
        else:
            input_np = y.transpose()

        bs = self.config.sep_data.test.batch_size
        seg_len = self.config.sep_data.test.seg_len
        win = np.repeat(np.hanning(seg_len+1)[:-1][::, np.newaxis], 2, axis=-1)
        all_len = input_np.shape[0]

        sep_list = list()
        #logits_list = list()
        final_res = np.zeros(input_np.shape)
        for st in tqdm(range(0, all_len, int(seg_len * 0.75))):
            ## omit last trunk
            if st+seg_len > all_len:
                break
            cnt_input_np = input_np[st:st+seg_len] * win
            cnt_input_np, silence = norm(cnt_input_np, thres=1e-2)
            if not silence: 
                cnt_input = torch.Tensor(cnt_input_np)
                cnt_input = cnt_input.unsqueeze(0)
                if self.half:
                    cnt_input = cnt_input.half().to(self.local_rank)
                else:
                    cnt_input = cnt_input.float().to(self.local_rank)
                with torch.no_grad():
                    try:
                        logits = self.model(cnt_input, sep_only=True)
                    except:
                        __import__('pdb').set_trace() 
                    #sep_result_np = sep_result.detach().cpu().squeeze(0).numpy()
                    logits_np = logits.detach().cpu().squeeze(0).numpy()
            else:
                logits_np = np.zeros(cnt_input_np.shape)
            final_res[st:st+seg_len] += logits_np
                ##TODO
            #sep_result_final = sep_result_np[-1]
            #sep_result_final = sep_result_final.transpose()
            #sep_list.append(sep_result_final)
            #logits_list.append(logits_np)
        # sep = np.concatenate(sep_list, axis=-1)
        #logits = np.concatenate(logits_list, axis=0)
        sf.write("/root/debug.wav", final_res, samplerate=44100)
        bgm = input_np - final_res
        sf.write("/root/debug_bgm.wav", bgm, samplerate=44100)
        __import__('pdb').set_trace() 

        __import__('pdb').set_trace() 
        # pred = np.argmax(logits, -1)
        # if batch['pitch_ref'] is not None:
        #    acc = np.mean(batch['pitch_ref'][:pred.shape[0]] == pred)
        #    return acc
            
        return None

        ## VIS
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure()
        ax1 = plt.subplot(2,1,1)
        librosa.display.specshow(librosa.amplitude_to_db(sep), y_axis='chroma', x_axis='time')
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        librosa.display.specshow(librosa.amplitude_to_db(input_np), y_axis='chroma', x_axis='time')
        plt.tight_layout()
        plt.savefig("/root/fuck.png")
        __import__('pdb').set_trace() 
    
    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
