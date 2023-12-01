import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import os
import pdb
import pprint
import wandb
from torchnet import meter


import sys
sys.path.append('../')
from LSTM_Resnet50_retrieval_model.args import get_args
from LSTM_Resnet50_retrieval_model.datasets import CoherenceDataset
from LSTM_Resnet50_retrieval_model.datasets import train_transform, val_transform
from LSTM_Resnet50_retrieval_model.networks import TextEncoder, ImageEncoder, DiscourseClassifier
from LSTM_Resnet50_retrieval_model.triplet_loss import global_loss, TripletLoss
from LSTM_Resnet50_retrieval_model.utils import param_counter, rank, get_lr, requires_grad, infinite_loader, compute_confidence_score

def create_model(args, device='cuda'):
    text_encoder = TextEncoder(
        emb_dim=args.word2vec_dim,
        hid_dim=args.rnn_hid_dim,
        z_dim=args.feature_dim,
        max_len = args.max_len,
        word2vec_file=f'models/word2vec_{args.data_source}.bin',
        with_attention=args.with_attention).to(device)
    image_encoder = ImageEncoder(
        z_dim=args.feature_dim).to(device)
    discourse_classifier = DiscourseClassifier(
        len(args.relations), args.feature_dim).to(device)

    # optimizer
    optimizer = torch.optim.Adam([
            {'params': text_encoder.parameters()},
            {'params': image_encoder.parameters()},
            {'params': discourse_classifier.parameters()},
        ], lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    print('# text_encoder', param_counter(text_encoder.parameters()))
    print('# image_encoder', param_counter(image_encoder.parameters()))
    print('# discourse_classifier', param_counter(discourse_classifier.parameters()))
    return text_encoder, image_encoder, discourse_classifier, optimizer, scheduler


def load_model(ckpt_path, device='cuda'):
    print('load from ckpt path:', ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    text_encoder, image_encoder, discourse_classifier, optimizer, scheduler = create_model(ckpt_args, device=device)
    
    text_encoder.load_state_dict(ckpt['text_encoder'])
    image_encoder.load_state_dict(ckpt['image_encoder'])
    discourse_classifier.load_state_dict(ckpt['discourse_classifier'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    batch_idx = ckpt['batch_idx']
    return ckpt_args, batch_idx, text_encoder, image_encoder, discourse_classifier, optimizer, scheduler


def save_model(args, batch_idx, text_encoder, image_encoder, discourse_classifier, optimizer, scheduler, ckpt_path):
    print('save ckpt to:', ckpt_path)
    ckpt = {
        'args': args,
        'batch_idx': batch_idx,
        'text_encoder': text_encoder.state_dict(),
        'image_encoder': image_encoder.state_dict(),
        'discourse_classifier': discourse_classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(ckpt, ckpt_path)


def train(args, train_loader, val_loader, text_encoder, image_encoder, discourse_classifier, crit_retrieval, crit_classification, optimizer, scheduler):
    train_loader = infinite_loader(train_loader)
    device = args.device
    valid_questions = get_valid_questions(args).to(device)
    pbar = tqdm(range(args.batch_idx_start, args.batch_idx_start+args.num_batches), smoothing=0.3)

    for batch_idx in pbar:
        batch = next(train_loader)
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        txt, txt_len, img, target = batch
        bs = img.shape[0]

        txt_feat, attn = text_encoder(txt.long(), txt_len)
        img_feat = image_encoder(img)
        output_class = discourse_classifier(txt_feat, img_feat)

        # pdb.set_trace()
        output_class = output_class[:, valid_questions]
        target = target[:, valid_questions]
        running_loss_classification = crit_classification(output_class, target)

        label = list(range(0, bs))
        label.extend(label)
        label = np.array(label)
        label = torch.tensor(label).long().to(device)
        running_loss_retrieval = global_loss(crit_retrieval, torch.cat((img_feat, txt_feat)), label)[0]

        running_loss = args.weight_retrieval*running_loss_retrieval + args.weight_classification*running_loss_classification
        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

        if args.wandb:
            logs = {}
            logs['running_loss_classification'] = running_loss_classification
            logs['running_loss_retrieval'] = running_loss_retrieval
            logs['running_loss'] = running_loss
            logs['lr'] = get_lr(optimizer)

        if batch_idx % args.val_freq == 0:
            requires_grad(text_encoder, False)
            requires_grad(image_encoder, False)
            requires_grad(discourse_classifier, False)
            text_encoder.eval()
            image_encoder.eval()
            discourse_classifier.eval()

            txt_feats = []
            img_feats = []
            probs = []
            preds = []
            labels = []
            for batch in tqdm(val_loader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)
                txt, txt_len, img, target = batch
                txt_feat, attn = text_encoder(txt.long(), txt_len)
                img_feat = image_encoder(img)
                prob = torch.sigmoid(discourse_classifier(txt_feat, img_feat))
                probs.append(prob)
                pred = prob > 0.5
                txt_feats.append(txt_feat.detach().cpu())
                img_feats.append(img_feat.detach().cpu())
                preds.append(pred.detach().cpu())
                labels.append(target)

            # retrieval metrics
            txt_feats = torch.cat(txt_feats, dim=0)
            img_feats = torch.cat(img_feats, dim=0)
            CIs = compute_confidence_score(txt_feats.to(device), img_feats.to(device), discourse_classifier, valid_questions)

            retrieved_range = min(txt_feats.shape[0], args.retrieved_range)
            MedR, recalls = rank(
                txt_feats.numpy(),
                img_feats.numpy(),
                CIs,
                retrieved_type=args.retrieved_type,
                retrieved_range=retrieved_range)
            scheduler.step(MedR.mean())

            # relation prediction
            probs = torch.cat(probs, dim=0).cpu().numpy()  # [N, valid_questions]
            preds = torch.cat(preds, dim=0).cpu().numpy()  # [N, valid_questions]
            labels = torch.cat(labels, dim=0).cpu().numpy()  # [N, valid_questions]

            # num_pred = (preds > 0).sum(1)
            # num_true = (labels > 0).sum(1)
            # print('num_pred = {:.2f}'.format(num_pred.mean()))
            # print('num_true = {:.2f}'.format(num_true.mean()))

            # f1, confusions = multilabel_acc(labels, preds)
            # for classidx in range(len(f1)):
            #     writer.add_scalar('f1-score-class-{}'.format(classidx), f1[classidx], epoch)
            # print('F1 scores: {}'.format(f1))
            # print('confusions:')
            # print(confusions)

            mtr = meter.APMeter()
            mtr.add(probs, labels)
            APs = mtr.value()
            mAP = APs.mean() # mean average precision

            if args.wandb:
                logs['MedR'] = MedR.mean()
                logs['MedR_std'] = MedR.std()
                for k, v in recalls.items():
                    logs[f'recall_{k}'] = v.mean()
                for q, ap in zip(args.questions.split(','), APs):
                    logs[f'AP_{q}'] = ap
                logs['mAP'] = mAP

                probs = probs[valid_questions.cpu().numpy()] # [N, valid_questions]
                entropies = - (probs * np.log(probs) + (1-probs) * np.log(1-probs)) # [N, valid_questions] # entropies
                confs = (1.0 / entropies).sum(axis=1) # [N]
                logs['average_conf'] = confs.mean()


            requires_grad(text_encoder, True)
            requires_grad(image_encoder, True)
            requires_grad(discourse_classifier, True)
            text_encoder.train()
            image_encoder.train()
            discourse_classifier.train()

        if args.wandb:
            wandb.log(logs)
        
        if batch_idx % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f'{batch_idx:>08d}.ckpt')
            save_model(args, batch_idx, text_encoder, image_encoder, discourse_classifier, optimizer, scheduler, ckpt_path)


def get_valid_questions(args):
    if args.dataset_q:  # ignore it for now
        valid_questions = torch.tensor([0])
    else:
        print(f'Use question(s): {args.questions}')
        if args.data_source == 'cite':
            valid_questions = torch.tensor([
                    int(x)-2 for x in args.questions.split(',')],
                    dtype=torch.long)
        else:
            valid_questions = torch.tensor([
                    int(x) for x in args.questions.split(',')],
                    dtype=torch.long)
    return valid_questions


if __name__ == '__main__':
    ##############################
    # setup
    ##############################
    args = get_args()
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(args.__dict__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    ##############################
    # datasets
    ##############################
    if args.data_source == 'cite':
        args.relations = ['q2_resp', 'q3_resp', 'q4_resp', 'q5_resp', 'q6_resp', 'q7_resp', 'q8_resp']
        args.max_len = 200
        args.questions = '2,3,4,5,6,7,8'
    else:
        args.relations = ['Visible', 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant']
        args.max_len = 40
        args.questions = '0,1,2,3,4,5'
    train_set = CoherenceDataset(
            part='train',
            datasource=args.data_source,
            word2vec_file=f'models/word2vec_{args.data_source}.bin',
            max_len=args.max_len,
            dataset_q=args.dataset_q,  # ignore it for now
            transform=train_transform)
    n2p = train_set.n2p

    test_set = CoherenceDataset(
            part='test',
            datasource=args.data_source,
            word2vec_file=f'models/word2vec_{args.data_source}.bin',
            max_len=args.max_len,
            dataset_q=args.dataset_q,  # ignore it for now
            transform=val_transform)

    if args.debug:
        train_set = torch.utils.data.Subset(train_set, indices=range(1000))
        test_set = torch.utils.data.Subset(test_set, indices=range(1000))

    train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
            drop_last=False)
    val_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=False)

    print('train data:', len(train_set), len(train_loader))
    print('test data:', len(test_set), len(val_loader))

    ##############################
    # model
    ##############################
    if args.ckpt_path:
        ckpt_args, batch_idx, text_encoder, image_encoder, discourse_classifier, optimizer, scheduler = load_model(args.ckpt_path, device)
        args.batch_idx_start = batch_idx + 1
    else:
        text_encoder, image_encoder, discourse_classifier, optimizer, scheduler = create_model(args, device)
        args.batch_idx_start = 0

    # criterion
    crit_retrieval = TripletLoss(margin=args.margin)

    if args.reweight:
        pos_weight = np.array(n2p)
        print('before reweight:', pos_weight)
        pos_weight[pos_weight > args.reweight_limit] = args.reweight_limit
        pos_weight[pos_weight < 1.0 / args.reweight_limit] = 1.0 / args.reweight_limit
        print('after reweight:', pos_weight)
        pos_weight = torch.FloatTensor(pos_weight)
        # valid_questions = get_valid_questions(args).to(device)
        # pos_weight = pos_weight[valid_questions]
        crit_classification = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    ##############################
    # train
    ##############################
    if args.wandb:
        project_name = 'coherence'
        wandb.init(project=project_name, config=args)
        wandb.config.update(args)
        save_dir = os.path.join(f'runs_{args.data_source}', wandb.run.id)
    else:
        from datetime import datetime
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.path.dirname(__file__), f'runs_{args.data_source}', time_stamp)

    img_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)
    args.save_dir = save_dir

    train(args, train_loader, val_loader, text_encoder, image_encoder, discourse_classifier, crit_retrieval, crit_classification, optimizer, scheduler)