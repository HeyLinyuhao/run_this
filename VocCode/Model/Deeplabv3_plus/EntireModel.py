import torch.nn
from Utils.losses import *
from itertools import chain
from Base.base_model import BaseModel
from Model.Deeplabv3_plus.encoder_decoder import *

res_net = "VocCode/Model/PSPNet/Backbones/pretrained/3x3resnet{}-imagenet.pth"
res_net_2 = "VocCode/Model/Deeplabv3_plus/Backbones/pretrained/resnet{}.pth"


class EntireModel(BaseModel):
    def __init__(self, num_classes, config, sup_loss=None, cons_w_unsup=None, ignore_index=None,
                 pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):
        super(EntireModel, self).__init__()
        self.encoder1 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                       pretrained_model=None, back_bone=config['resnet'])
        self.decoder1 = DecoderNetwork(num_classes=num_classes, data_shape=config['data_h_w'],cls_specific_recon=False, num_semantic_cls=None)
        self.encoder2 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                       pretrained_model=None, back_bone=config['resnet'])
        self.decoder2 = DecoderNetwork(num_classes=num_classes, data_shape=config['data_h_w'])
        self.encoder_s = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                        pretrained_model=res_net_2.format(str(config['resnet'])),
                                        back_bone=config['resnet'])
        self.decoder_s = DecoderNetwork(num_classes=num_classes, data_shape=config['data_h_w'],cls_specific_recon=False, num_semantic_cls=None)

        self.decoder_sa = DecoderNetwork(num_classes=3, data_shape=config['data_h_w'],cls_specific_recon=True, num_semantic_cls=21)


        self.mode = "semi" if config['semi'] else "sup"
        self.sup_loss = sup_loss
        self.ignore_index = ignore_index
        self.unsup_loss_w = cons_w_unsup
        self.unsuper_loss = semi_ce_loss

    def freeze_teachers_parameters(self):
        for p in self.encoder1.parameters():
            p.requires_grad = False
        for p in self.decoder1.parameters():
            p.requires_grad = False

        for p in self.encoder2.parameters():
            p.requires_grad = False
        for p in self.decoder2.parameters():
            p.requires_grad = False

    def warm_up_forward(self, id, x, y):
        if id == 1:
            output_l = self.decoder1(self.encoder1(x))['pred']
        elif id == 2:
            output_l = self.decoder2(self.encoder2(x))['pred']
        else:
            output_l = self.decoder_s(self.encoder_s(x))['pred']

        loss = F.cross_entropy(output_l, y, ignore_index=self.ignore_index)
        curr_losses = {'loss_sup': loss}
        outputs = {'sup_pred': output_l}
        return loss, curr_losses, outputs

    def forward(self,x_cut = None, x_l=None, target_l=None, x_ul=None, x_ul_ori=None, target_ul=None, curr_iter=None, epoch=None, id=0,
                warm_up=False, image_level_label=None, semi_p_th=0.6, semi_n_th=0.0):
        if warm_up:
            return self.warm_up_forward(id=id, x=x_l, y=target_l)


        enc_l = self.encoder_s(x_l)

        output_l = self.decoder_s(enc_l)['pred']

        # mask = torch.argmax(target_l, dim=1)



        output_l2 = self.decoder_sa(enc_l)['pred']
        # print(len(output_l2)) 
        # 21
        # print(output_l2[0].shape) 
        # [8,3,w,h]

        rec_loss = 0
        count = 0
        
        for b in range(x_l.shape[0]):
            for i in range(21):

                mask = (target_l[b]!=i)
                mask_op = (target_l[b]==i)
                # print('mask')
                # print(mask.shape) [w,h]
                # if torch.sum(mask_op)!=0:
                target_l2 = x_l[b]*mask_op
                count+=1
                rec_loss+=F.mse_loss(output_l2[i][b], target_l2)
        
        all_pred=output_l2[0]
        for i in range(20):
            all_pred = torch.add(all_pred,output_l2[i+1])

        
        rec_loss/=count
        rec_loss+=F.mse_loss(all_pred, x_l)
        # torch.sum(output_l2, 0)
        loss_sup = F.cross_entropy(output_l, target_l, ignore_index=self.ignore_index) 


        curr_losses = {'loss_sup': loss_sup}


        enc_ul = self.encoder_s(x_ul)
        output_ul = self.decoder_s(enc_ul)['pred']

        # enc_ul_ori = self.encoder_s(x_ul)
        # output_ul2 = self.decoder_sa(enc_ul_ori)['pred']  

        # all_pred_ul=output_ul2[0]
        # for i in range(20):
        #     all_pred_ul = torch.add(all_pred_ul,output_ul2[i+1])

        
        # rec_loss+=F.mse_loss(all_pred_ul, x_ul)
        
        # unsupervised loss
        # in our experiments, negative learning doesn't bring consist improvements
        # fix to be 0 in cmd arguments
        loss_unsup, pass_rate, neg_loss = self.unsuper_loss(inputs=output_ul, targets=target_ul,
                                                            conf_mask=True, threshold=semi_p_th,
                                                            threshold_neg=semi_n_th)

        confident_reg = .1 * torch.mean(F.softmax(target_ul, dim=1) ** 2)

        if semi_n_th > .0:
            loss_unsup += neg_loss
            loss_unsup += confident_reg

        loss_unsup = loss_unsup * self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
        
        
        # loss_rec = F.mse_loss(output_l2, target_l2)
        total_loss = loss_unsup + loss_sup + 0.1*rec_loss

        curr_losses['loss_unsup'] = loss_unsup
        curr_losses['loss_rec'] = rec_loss
        curr_losses['pass_rate'] = pass_rate
        curr_losses['neg_loss'] = neg_loss
        outputs = {'sup_pred': output_l, 'unsup_pred': output_ul,'lab_img': x_l,'lab_rec_img': all_pred}

        # outputs = {'sup_pred': output_l, 'unsup_pred': output_ul,'lab_img': x_l,'unlab_img': x_ul,'lab_rec_img': all_pred,'unlab_rec_img': all_pred_ul }
        return total_loss, curr_losses, outputs


