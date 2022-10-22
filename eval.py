import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import    Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric
from data import reOrganize_mDATA


parser = argparse.ArgumentParser(description='abc')
testMask_dir = '' ## Point to the Camelyon test set mask location

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
parser.add_argument('--train_csv',type=str)   ## MaxMinS, MaxS, AFS
parser.add_argument('--val_csv',type=str)   ## MaxMinS, MaxS, AFS
parser.add_argument('--test_csv',type=str)   ## MaxMinS, MaxS, AFS

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)

def main():
    params = parser.parse_args()
    epoch_step = json.loads(params.epoch_step)
    model_dict = torch.load("./debug_log/fold5/best_model.pth")

    in_chn = 1024

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    classifier.load_state_dict(model_dict["classifier"])
    attention = Attention(params.mDim).to(params.device)
    attention.load_state_dict(model_dict["attention"])
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    dimReduction.load_state_dict(model_dict["dim_reduction"])
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)
    attCls.load_state_dict(model_dict["att_classifier"])

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)

    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log_test.txt')
    #save_dir = os.path.join(params.log_dir, 'best_model.pth')
    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))

    # with open(params.mDATA0_dir_train0, 'rb') as f:
    #     mDATA_train = pickle.load(f)
    # with open(params.mDATA0_dir_val0, 'rb') as f:
    #     mDATA_val = pickle.load(f)
    # with open(params.mDATA_dir_test0, 'rb') as f:
    #     mDATA_test = pickle.load(f)

    mDATA_train = params.train_csv
    mDATA_val = params.val_csv
    mDATA_test = params.test_csv

    #SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train)
    #SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val)
    SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA(mDATA_test)


    tauc = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                    UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=0,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test, total_instance=params.total_instance_test, distill=params.distill_type)


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, criterion=None,  params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst].to(params.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0, pr0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1, pr1 = eval_metric(gPred_1, gt_1)

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}, PR {pr0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}, PR {pr1}', f_log)

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1


def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, epoch, ce_cri=None, params=None,
                                          f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):

    SlideNames_list, mFeat_list, Label_dict = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)
    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(params.device))
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            # optimizer0.step()

            ## optimization for the second tier
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer0.step()
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} '
            print_log(tstr, f_log)

    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


if __name__ == "__main__":
    main()
