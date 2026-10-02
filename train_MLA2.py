# import argparse
# import os
# import ast
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import pdb

# # from dataset.dataset import AVDataset, CAVDataset, M3AEDataset, TVDataset, Modal3Dataset, CLIPDataset

# from dataset.dataloader import AV_CD_Dataset
# # from model.basic_model import VA_Classifier,TVA_Classifier 
# from model.basic_model import VA_MLA_Classifier,TVA_MLA_Classifier
# # from model.AVClass import AVClassifier
# from dataset.Mydataset import TVADataset
# from dataset.Mydataset import M3AEDataset
# from dataset.Mydataset import CramedDataset,AVEDataset,KSDataset

# # from models.basic_model import AVClassifier, CAVClassifier, M3AEClassifier, Modal3Classifier, CLIPClassifier
# from utils.metrics import calculate_metrics
# from utils.utils import setup_seed,weight_init,print_model_params,print_current_lrs
# from utils.utils import Alignment,getAlpha_Learnable_Fitted
# from utils.utils_MLA import GSPlugin,calculate_gating_weights,calculate_gating_weights3
# import datetime

# def get_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default="CREMA-D", type=str,
#                         help='Currently, we only support Food-101, MVSA, CREMA-D')
#     parser.add_argument('--modulation', default='Normal', type=str,
#                         choices=['Normal', 'OGM', 'OGM_GE', "QMF"])
#     parser.add_argument('--fusion_method', default='concat', type=str,
#                         choices=['sum', 'concat', 'gated', 'film'])
#     parser.add_argument('--fps', default=1, type=int)
#     parser.add_argument('--use_video_frames', default=3, type=int)
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--epochs', default=100, type=int)

#     parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
#     parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
#     parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
#     parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

#     parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
#     parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
#     parser.add_argument('--alpha', default = 0.3, type=float, help='alpha in OGM-GE')

#     parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
#     parser.add_argument('--train', action='store_true', help='turn on train mode')

#     parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
#     parser.add_argument('--tensorboard_path', default = "ckpt/", type=str, help='path to save tensorboard logs')

#     parser.add_argument('--random_seed', default=42, type=int)
#     parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
#     parser.add_argument('--lorb', default="m3ae", type=str, help='model_select in [large, base, m3ae]')
#     parser.add_argument('--gs_flag', action='store_true')
#     parser.add_argument('--av_alpha', default=0.5, type=float, help='2 modal fusion alpha in GS')
#     parser.add_argument('--cav_opti', action='store_true')
#     parser.add_argument('--cav_lrs', action='store_true')
#     parser.add_argument('--cav_augnois', action='store_true')
#     parser.add_argument('--modal3', action='store_true', help='3 modality fusion flag')
#     parser.add_argument('--dynamic', action='store_true', help='if dynamic fusion in GS')
#     parser.add_argument('--a_alpha', default=0.35, type=float, help='audio alpha in 3 modal GS')
#     parser.add_argument('--v_alpha', default=0.25, type=float, help='visual alpha in 3 modal GS')
#     parser.add_argument('--t_alpha', default=0.4, type=float, help='textual alpha in 3 modal GS')
#     parser.add_argument('--clip', action='store_true', help='run using clip pre-trained feature')
#     parser.add_argument('--ckpt_load_path_train', default = None, type=str, help='loaded path when training')
    
#     parser.add_argument('--Use_initWeight', default=False, type=bool, help='Use weight init model')
#     parser.add_argument('--model_name', default='["Visual","Audio"]', type=str, choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
#     parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
#     parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')

      
#     return parser.parse_args()

# def calculate_entropy(output):
#     probabilities = F.softmax(output, dim=0)
#     # probabilities = F.softmax(output, dim=1)
#     log_probabilities = torch.log(probabilities)
#     entropy = -torch.sum(probabilities * log_probabilities)
#     return entropy

# def calculate_gating_weights(encoder_output_1, encoder_output_2):
    
#     entropy_1 = calculate_entropy(encoder_output_1)
#     entropy_2 = calculate_entropy(encoder_output_2)
    
#     max_entropy = max(entropy_1, entropy_2)
    
#     gating_weight_1 = torch.exp(max_entropy - entropy_1)
#     gating_weight_2 = torch.exp(max_entropy - entropy_2)
    
#     sum_weights = gating_weight_1 + gating_weight_2
    
#     gating_weight_1 /= sum_weights
#     gating_weight_2 /= sum_weights
    
#     return gating_weight_1, gating_weight_2

# def calculate_gating_weights3(encoder_output_1, encoder_output_2, encoder_output_3):
#     entropy_1 = calculate_entropy(encoder_output_1)
#     entropy_2 = calculate_entropy(encoder_output_2)
#     entropy_3 = calculate_entropy(encoder_output_3)
    
#     max_entropy = max(entropy_1, entropy_2, entropy_3)
    
#     gating_weight_1 = torch.exp(max_entropy - entropy_1)
#     gating_weight_2 = torch.exp(max_entropy - entropy_2)
#     gating_weight_3 = torch.exp(max_entropy - entropy_3)
    
#     sum_weights = gating_weight_1 + gating_weight_2 + gating_weight_3
    
#     gating_weight_1 /= sum_weights
#     gating_weight_2 /= sum_weights
#     gating_weight_3 /= sum_weights
    
#     return gating_weight_1, gating_weight_2, gating_weight_3

# def rank_loss(confidence, idx, history):
#     # make input pair
#     rank_input1 = confidence
#     rank_input2 = torch.roll(confidence, -1)
#     idx2 = torch.roll(idx, -1)

#     # calc target, margin
#     rank_target, rank_margin = history.get_target_margin(idx, idx2)
#     rank_target_nonzero = rank_target.clone()
#     rank_target_nonzero[rank_target_nonzero == 0] = 1
#     rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

#     # ranking loss
#     ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
#                                         rank_input2,
#                                         -rank_target.reshape(-1,1))

#     return ranking_loss

# def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, 
#                 gs_plugin = None, writer=None, gs_flag = False, av_alpha = 0.5,
#                 txt_history = None, img_history = None, audio_history = None):
#     criterion = nn.CrossEntropyLoss()
#     softmax = nn.Softmax(dim=1)
#     relu = nn.ReLU(inplace=True)
#     tanh = nn.Tanh()
#     if gs_plugin is None:
#         gs_plugin = GSPlugin()
#     model.train()
#     print("Start training ... ")
#     modal_names = ast.literal_eval(args.model_name)
#     _loss = 0
#     _loss_a = 0
#     _loss_v = 0
#     _loss_t = 0
#     len_dataloader = len(dataloader)
#     print(f"modal_names is {modal_names}")
#     for batch_step, data_packet in enumerate(dataloader):
#         if modal_names == ["Visual", "Audio"]:
#             spec, image,  label  = data_packet[0],data_packet[1],data_packet[2]
#             spec, image, label = spec.to(device), image.to(device), label.to(device)
#             if args.dataset == 'CREMAD':
#                 data_mini_packet = (image.float(), spec.float())
#             else:
#                 data_mini_packet = (image.float(), spec.unsqueeze(1).float())
#         elif modal_names == ["Image", "Text"]:
#             token, padding_mask, image, label, _ = data_packet
#             token, padding_mask = token.to(device), padding_mask.to(device)
#             image, label = image.to(device), label.to(device)
#             data_mini_packet = (token, padding_mask, image)
#         elif modal_names == ["Text", "Visual", "Audio"]:
#             token, padding_mask, image, spec, label, _ = data_packet
#             token, padding_mask = token.to(device), padding_mask.to(device)
#             image, spec, label = image.to(device), spec.to(device), label.to(device)
#             data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
#         else:
#             raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
#         # 对第一个模态进行GS训练
#         if len(modal_names) == 2:
#             v,a = model(data_mini_packet)
#         else:
#             t,v,a = model(data_mini_packet)
#         out_a = model.fusion_model.fc_out(a)
#         loss_a = criterion(out_a, label)
#         loss_a.backward()

#         gs_plugin.before_update(model.fusion_model.fc_out, a, 
#                                 batch_step, len_dataloader, gs_plugin.exp_count)
#         optimizer.step()
#         optimizer.zero_grad()

#         gs_plugin.exp_count += 1
#         # 对第二个模态进行GS训练
#         out_v = model.fusion_model.fc_out(v)
        
#         loss_v = criterion(out_v, label)
#         loss_v.backward()

#         gs_plugin.before_update(model.fusion_model.fc_out, v, 
#                                 batch_step, len_dataloader, gs_plugin.exp_count)
#         optimizer.step()
#         optimizer.zero_grad()

#         gs_plugin.exp_count += 1
#         # 对第三个模态进行GS训练
#         if args.modal3:
#             out_t = model.fusion_model.fc_out(t)
            
#             loss_t = criterion(out_t, label)
#             loss_t.backward()

#             gs_plugin.before_update(model.fusion_model.fc_out, t, 
#                                     batch_step, len_dataloader, gs_plugin.exp_count)
#             optimizer.step()
#             optimizer.zero_grad()

#             gs_plugin.exp_count += 1

#         for n, p in model.named_parameters():
#             if p.grad != None:
#                 del p.grad

#         _loss += (loss_a * av_alpha + loss_v * (1 - av_alpha)).item()
#         _loss_a += loss_a.item()
#         _loss_v += loss_v.item()
#         if args.modal3:
#             _loss_t += loss_t.item()

        
#     scheduler.step()
#     if args.modal3:
#         return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _loss_t / len(dataloader)    
#     return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)

# def valid(args, model, device, dataloader, 
#           gs_flag = False, av_alpha = 0.5, 
#           a_alpha = 0.35, v_alpha = 0.25, t_alpha = 0.4):
#     softmax = nn.Softmax(dim=1)
#     modal_names = ast.literal_eval(args.model_name)
#     n_classes = args.num_classes
#     with torch.no_grad():
#         model.eval()
#         num = [0.0 for _ in range(n_classes)]
#         acc = [0.0 for _ in range(n_classes)]
#         acc_a = [0.0 for _ in range(n_classes)]
#         acc_v = [0.0 for _ in range(n_classes)]
#         acc_t = [0.0 for _ in range(n_classes)]
#         pred_result = []
#         for step, data_packet in enumerate(dataloader):
#             # 加载数据
#             if modal_names == ["Visual", "Audio"]:
#                 spec, image,label  = data_packet[0],data_packet[1],data_packet[2]
#                 # spec, image, label, _ = batch
#                 spec, image, label = spec.to(device), image.to(device), label.to(device)
#                 if args.dataset == 'CREMAD':
#                     data_mini_packet = (image.float(), spec.float())
#                 else:
#                     data_mini_packet = (image.float(), spec.unsqueeze(1).float())
#             elif modal_names == ["Image", "Text"]:
#                 token, padding_mask, image, label, _ = data_packet
#                 token, padding_mask = token.to(device), padding_mask.to(device)
#                 image, label = image.to(device), label.to(device)
#                 data_mini_packet = (token, padding_mask, image)
#             elif modal_names == ["Text", "Visual", "Audio"]:
#                 token, padding_mask, image, spec, label, _ = data_packet
#                 token, padding_mask = token.to(device), padding_mask.to(device)
#                 image, spec, label = image.to(device), spec.to(device), label.to(device)
#                 data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
#              # 对第一个模态进行GS训练
#             if len(modal_names) == 2:
#                 v,a = model(data_mini_packet)
#             else:
#                 t,v,a = model(data_mini_packet)
#             # out_a = model.fusion_model.fc_out(a)
#             # out_v = model.fusion_model.fc_out(v)
#             out_a = model.fusion_model.fc_out(a)
#             out_v = model.fusion_model.fc_out(v)
#             if len(modal_names) == 3:
#                 out_t = model.fusion_model.fc_out(t)
#             if args.dynamic:
#                 if args.modal3:
#                     audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
#                     out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)
#                 else:
#                     txt_conf, img_conf = calculate_gating_weights(out_a, out_v)
#                     out = (out_a * txt_conf + out_v * img_conf)
#             else:
#                 if args.modal3:
#                     out = a_alpha * out_a + v_alpha * out_v + t_alpha * out_t
#                 else:
#                     out = av_alpha * out_a + (1-av_alpha) * out_v

#             prediction = softmax(out)
#             pred_v = softmax(out_v)
#             pred_a = softmax(out_a)
#             if args.modal3:
#                 pred_t = softmax(out_t)

#             for i in range(image.shape[0]):

#                 ma = np.argmax(prediction[i].cpu().data.numpy())
#                 v = np.argmax(pred_v[i].cpu().data.numpy())
#                 a = np.argmax(pred_a[i].cpu().data.numpy())
#                 if args.modal3:
#                     t = np.argmax(pred_t[i].cpu().data.numpy())
#                 num[label[i]] += 1.0

#                 if np.asarray(label[i].cpu()) == ma:
#                     acc[label[i]] += 1.0
#                 if np.asarray(label[i].cpu()) == v:
#                     acc_v[label[i]] += 1.0
#                 if np.asarray(label[i].cpu()) == a:
#                     acc_a[label[i]] += 1.0
#                 if args.modal3:
#                     if np.asarray(label[i].cpu()) == t:
#                         acc_t[label[i]] += 1.0
#     if args.modal3:
#         return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num)    
#     return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

# # average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
# def wa_model(exp_dir):
#     all_ckpts = os.listdir(exp_dir)
#     sdA = torch.load(os.path.join(exp_dir, all_ckpts[0]), map_location='cpu')["model"]
#     model_cnt = 1
#     for epoch in range(1, len(all_ckpts)):
#         sdB = torch.load(os.path.join(exp_dir, all_ckpts[epoch]), map_location='cpu')["model"]
#         for key in sdA:
#             sdA[key] = sdA[key] + sdB[key]
#         model_cnt += 1
#     print('wa {:d} models from {:d} to {:d}'.format(model_cnt, 1, len(all_ckpts)))
#     for key in sdA:
#         sdA[key] = sdA[key] / float(model_cnt)
#     return sdA


# def main(av_alpha = 0.5):
#     args = get_arguments()
#     # print(args)

#     setup_seed(args.random_seed)
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
#     gpu_ids = list(range(torch.cuda.device_count()))

#     device = torch.device('cuda:0')
#     # ==================数据集加载=====================================
#     if args.dataset == 'CREMAD':
#         args.num_classes = 6
#         # train_dataset = CramedDataset(mode='train', args=args)
#         # test_dataset = CramedDataset(mode='test', args=args)
#         train_dataset = AV_CD_Dataset(mode='train')
#         test_dataset = AV_CD_Dataset(mode='test')
#         print(f"use train AV_CD_Dataset and test AV_CD_Dataset for CREMAD dataset")
#     elif args.dataset == 'KineticSound':
#         args.num_classes = 34
#         train_dataset = KSDataset(mode='train', args=args)
#         test_dataset = KSDataset(mode='test', args=args)
#     elif args.dataset == 'AVE':
#         args.num_classes = 28 
#         train_dataset = AVEDataset(mode='train', args=args)
#         test_dataset = AVEDataset(mode='test', args=args)
#     elif args.dataset == 'Food101':
#         args.num_classes = 101
#         train_dataset = M3AEDataset(args,mode='train')
#         test_dataset = M3AEDataset(args,mode='test')
#     elif args.dataset == 'MVSA':
#         args.num_classes = 3
#         train_dataset = M3AEDataset(args,mode='train')
#         test_dataset = M3AEDataset(args,mode='test')
#     elif args.dataset == 'IEMOCAP3':
#         args.num_classes = 5
#         train_dataset = TVADataset(mode='train', args=args, pick_num=3)
#         test_dataset = TVADataset(mode='test', args=args, pick_num=3)
#     else:
#         raise NotImplementedError('Incorrect dataset name {}! '
#                                   'Only support AVE, KineticSound and CREMA-D for now!'.format(args.dataset))
#     print("train_dataset size: {}".format(len(train_dataset)))
#     print("test_dataset size: {}".format(len(test_dataset)))
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


#     if args.model_name == '["Visual","Audio"]' or args.model_name == '["Image","Text"]':
#         print(f"Using {args.model_name} model for training.")
#         model = VA_MLA_Classifier(args)
#     else:
#         model = TVA_MLA_Classifier(args)
#     if args.ckpt_load_path_train:
#         loaded_dict = torch.load(args.ckpt_load_path_train)
#         state_dict = loaded_dict['model']
#         state_dict = {key[7:]: state_dict[key] for key in state_dict}
#         del state_dict["fusion_module.fc_out.weight"]
#         del state_dict["fusion_module.fc_out.bias"]
#         missing, unexcepted = model.load_state_dict(state_dict, strict = False)
#         print('Trained model loaded!')
    
#     model.to(device)
#     print_model_params(model)
#     # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
#     model.cuda()
#     if args.model_name == '["Visual","Audio"]' and args.Use_initWeight:
#         model.apply(weight_init)
#         print(f"Use Weight int")

#     if args.lorb == "large" and args.cav_opti:
#         # optimizer = optim.SGD(model.fusion_model.fc_out.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#         # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#         mlp_list = ['fusion_module.fc_out.weight', 'module.fusion_module.fc_out.bias']
#         mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
#         base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
#         mlp_params = [i[1] for i in mlp_params]
#         base_params = [i[1] for i in base_params]
#         optimizer = optim.Adam([{'params': base_params, 'lr': args.learning_rate / 10}, 
#                                 {'params': mlp_params, 'lr': args.learning_rate}],
#                                 weight_decay=5e-7, 
#                                 betas=(0.95, 0.999))
#     else:
#         optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#         # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 0.0, betas=(0.9, 0.999))
#     if args.lorb == "large" and args.cav_lrs:
#         args.lrscheduler_start = 2
#         args.lrscheduler_step = 1
#         args.lrscheduler_decay = 0.5
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                     list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
#                                                     gamma = args.lrscheduler_decay)    

#     else:
#         scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

   
#     # GS Plugin
#     gs = GSPlugin()
#     txt_history = None
#     img_history = None
#     audio_history = None    
#     print(f"args num_class is {args.num_classes}")
#     if args.train:

#         best_acc = 0.0
#         if args.gs_flag:
#             log_name = '{}_{}_{}'.format(args.fusion_method, "GS", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         else:
#             log_name = '{}_{}_{}'.format(args.fusion_method, args.modulation, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         for epoch in range(args.epochs):

#             print('Epoch: {}: '.format(epoch))

#             if args.use_tensorboard:

#                 writer_path = os.path.join(args.tensorboard_path, args.dataset, log_name)
#                 if not os.path.exists(writer_path):
#                     os.mkdir(writer_path)
#                 writer = SummaryWriter(writer_path)

#                 if args.modal3:
#                     batch_loss, batch_loss_a, batch_loss_v, batch_loss_t = train_epoch(args, epoch, model, device, 
#                                                                      train_dataloader, optimizer,
#                                                                      scheduler, gs_plugin = gs, 
#                                                                      writer = writer, 
#                                                                      gs_flag = args.gs_flag, 
#                                                                      av_alpha = av_alpha,
#                                                                      txt_history = txt_history,
#                                                                      img_history = img_history,
#                                                                      audio_history=audio_history)
#                     acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
#                                             av_alpha= av_alpha, 
#                                             gs_flag= args.gs_flag,
#                                             a_alpha= args.a_alpha,
#                                             v_alpha= args.v_alpha,
#                                             t_alpha= args.t_alpha)

#                     writer.add_scalars('Loss', {'Total Loss': batch_loss,
#                                                 'Audio Loss': batch_loss_a,
#                                                 'Visual Loss': batch_loss_v,
#                                                 'Text Loss': batch_loss_t}, epoch)

#                     writer.add_scalars('Evaluation', {'Total Accuracy': acc,
#                                                     'Audio Accuracy': acc_a,
#                                                     'Visual Accuracy': acc_v,
#                                                     'Text Accuracy': acc_t}, epoch)
#                 else:
#                     print(gs)
#                     batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device, 
#                                                                         train_dataloader, optimizer,
#                                                                         scheduler, gs_plugin = gs, 
#                                                                         writer = writer, 
#                                                                         gs_flag = args.gs_flag, 
#                                                                         av_alpha = av_alpha,
#                                                                         txt_history = txt_history,
#                                                                         img_history = img_history)
#                     acc, acc_a, acc_v = valid(args, model, device, test_dataloader, 
#                                             av_alpha= av_alpha, 
#                                             gs_flag= args.gs_flag)

#                     writer.add_scalars('Loss', {'Total Loss': batch_loss,
#                                                 'Audio Loss': batch_loss_a,
#                                                 'Visual Loss': batch_loss_v}, epoch)

#                     writer.add_scalars('Evaluation', {'Total Accuracy': acc,
#                                                     'Audio Accuracy': acc_a,
#                                                     'Visual Accuracy': acc_v}, epoch)

#             else:
#                 batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
#                                                                      train_dataloader, optimizer, scheduler)
#                 acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

#             if acc > best_acc:
#                 best_acc = float(acc)

#                 if not os.path.exists(args.ckpt_path):
#                     os.mkdir(args.ckpt_path)

#                 model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
#                              'optimizer_{}_modulate_starts_{}_ends_{}_' \
#                              'epoch_{}_acc_{}.pth'.format(args.dataset,
#                                                           args.modulation,
#                                                           args.alpha,
#                                                           args.optimizer,
#                                                           args.modulation_starts,
#                                                           args.modulation_ends,
#                                                           epoch, acc)

#                 saved_dict = {'saved_epoch': epoch,
#                               'modulation': args.modulation,
#                               'alpha': args.alpha,
#                               'fusion': args.fusion_method,
#                               'acc': acc,
#                               'model': model.state_dict(),
#                               'optimizer': optimizer.state_dict(),
#                               'scheduler': scheduler.state_dict()}

#                 save_dir = os.path.join(args.ckpt_path, model_name)

#                 torch.save(saved_dict, save_dir)
#                 print('The best model has been saved at {}.'.format(save_dir))
#                 print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
#                 if args.modal3:
#                     print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
#                 else:    
#                     print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
#             else:
#                 print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
#                 if args.modal3:
#                     print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
#                 else:    
#                     print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))

#     else:
#         # if args.lorb == "large":
#         #     state_dict = wa_model("ckpt/")
#         # else:
#         # first load trained model
#         loaded_dict = torch.load(args.ckpt_path)
#         # epoch = loaded_dict['saved_epoch']
#         modulation = loaded_dict['modulation']
#         # alpha = loaded_dict['alpha']
#         fusion = loaded_dict['fusion']
#         state_dict = loaded_dict['model']

#         missing, unexcepted = model.load_state_dict(state_dict)
#         print('Trained model loaded!')
        
#         if not args.modal3:
#             acc, acc_a, acc_v = valid(args, model, device, 
#                                       test_dataloader, args.ewc_flag, args.gs_flag, args.av_alpha)
#             print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
#         else:
#             acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
#                                              args.ewc_flag, args.gs_flag, args.av_alpha,
#                                              a_alpha= args.a_alpha, v_alpha= args.v_alpha, t_alpha= args.t_alpha)
#             print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}, accuracy_t: {}'.format(acc, acc_a, acc_v, acc_t))


# if __name__ == "__main__":
#     main(av_alpha = 0.55)
import argparse
import os
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import tqdm
import time

# from dataset.dataset import AVDataset, CAVDataset, M3AEDataset, TVDataset, Modal3Dataset, CLIPDataset

from dataset.dataloader import AV_CD_Dataset
# from model.basic_model import VA_Classifier,TVA_Classifier 
from model.basic_model import VA_MLA_Classifier,TVA_MLA_Classifier
# from model.AVClass import AVClassifier
from dataset.Mydataset import TVADataset
from dataset.Mydataset import M3AEDataset
from dataset.Mydataset import CramedDataset,AVEDataset,KSDataset

# from models.basic_model import AVClassifier, CAVClassifier, M3AEClassifier, Modal3Classifier, CLIPClassifier
from utils.metrics import calculate_metrics
from utils.utils import setup_seed,weight_init,print_model_params,print_current_lrs
from utils.utils import Alignment,getAlpha_Learnable_Fitted
from utils.utils_MLA import GSPlugin,calculate_gating_weights,calculate_gating_weights3
import datetime

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="CREMA-D", type=str,
                        help='Currently, we only support Food-101, MVSA, CREMA-D')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', "QMF"])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default = 0.3, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default = "ckpt/", type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--lorb', default="m3ae", type=str, help='model_select in [large, base, m3ae]')
    parser.add_argument('--gs_flag', action='store_true')
    parser.add_argument('--av_alpha', default=0.5, type=float, help='2 modal fusion alpha in GS')
    parser.add_argument('--cav_opti', action='store_true')
    parser.add_argument('--cav_lrs', action='store_true')
    parser.add_argument('--cav_augnois', action='store_true')
    parser.add_argument('--modal3', action='store_true', help='3 modality fusion flag')
    parser.add_argument('--dynamic', action='store_true', help='if dynamic fusion in GS')
    parser.add_argument('--a_alpha', default=0.35, type=float, help='audio alpha in 3 modal GS')
    parser.add_argument('--v_alpha', default=0.25, type=float, help='visual alpha in 3 modal GS')
    parser.add_argument('--t_alpha', default=0.4, type=float, help='textual alpha in 3 modal GS')
    parser.add_argument('--clip', action='store_true', help='run using clip pre-trained feature')
    parser.add_argument('--ckpt_load_path_train', default = None, type=str, help='loaded path when training')
    
    parser.add_argument('--Use_initWeight', default=False, type=bool, help='Use weight init model')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str, choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')

      
    return parser.parse_args()

def calculate_entropy(output):
    # Notice!
    # For a batch, it should use dim = 1
    probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    # per-sample entropy, shape (B,)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy


def calculate_gating_weights(logit_map: dict):
    # Refactored gating that accepts a dict of logits {name: tensor(B, C)}
    name_list = list(logit_map.keys())

    e_map = {}
    w_map = {}

    for name in name_list:
        # per-sample entropy -> (B,) then make (B,1)
        e_map[name] = calculate_entropy(logit_map[name]).unsqueeze(1)

    combined_entropy = torch.cat([e_map[n] for n in name_list], dim=1)  # (B, M)
    max_entropy = torch.max(combined_entropy, dim=1)
    max_entropy = max_entropy.values.unsqueeze(1)  # (B,1)

    for name in name_list:
        w_map[name] = torch.exp(max_entropy - e_map[name])

    sum_weights = sum([w_map[n] for n in name_list])

    for name in name_list:
        w_map[name] = w_map[name] / sum_weights

    return w_map, e_map
    
def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target.reshape(-1,1))

    return ranking_loss

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, 
                gs_plugin = None, writer=None, gs_flag = False, av_alpha = 0.5,
                txt_history = None, img_history = None, audio_history = None):
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    if gs_plugin is None:
        gs_plugin = GSPlugin()
    model.train()
    print("Start training ... ")
    modal_names = ast.literal_eval(args.model_name)
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _loss_t = 0
    len_dataloader = len(dataloader)
    print(f"modal_names is {modal_names}")
    # 使用 tqdm 显示 batch 进度
    data_iter = tqdm.tqdm(enumerate(dataloader), total=len_dataloader, desc=f"Epoch {epoch}", ncols=120)
    for batch_step, data_packet in data_iter:
        if modal_names == ["Visual", "Audio"]:
            spec, image,  label  = data_packet[0],data_packet[1],data_packet[2]
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            if args.dataset == 'CREMAD':
                data_mini_packet = (image.float(), spec.float())
            else:
                data_mini_packet = (image.float(), spec.unsqueeze(1).float())
        elif modal_names == ["Image", "Text"]:
            token, padding_mask, image, label, _ = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, label = image.to(device), label.to(device)
            data_mini_packet = (token, padding_mask, image)
        elif modal_names == ["Text", "Visual", "Audio"]:
            token, padding_mask, image, spec, label, _ = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, spec, label = image.to(device), spec.to(device), label.to(device)
            data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
        else:
            raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
        # 对第一个模态进行GS训练
        if len(modal_names) == 2:
            v,a = model(data_mini_packet)
        else:
            t,v,a = model(data_mini_packet)
        out_a = model.fusion_model.fc_out(a)
        loss_a = criterion(out_a, label)*args.alpha
        loss_a.backward()

        gs_plugin.before_update(model.fusion_model.fc_out, a, 
                                batch_step, len_dataloader, gs_plugin.exp_count)
        optimizer.step()
        optimizer.zero_grad()

        gs_plugin.exp_count += 1
        
        # 对第二个模态进行GS训练
        out_v = model.fusion_model.fc_out(v)
        loss_v = criterion(out_v, label)*args.alpha
        loss_v.backward()

        gs_plugin.before_update(model.fusion_model.fc_out, v, 
                                batch_step, len_dataloader, gs_plugin.exp_count)
        optimizer.step()
        optimizer.zero_grad()

        gs_plugin.exp_count += 1
        # 对第三个模态进行GS训练
        if args.modal3:
            out_t = model.fusion_model.fc_out(t)
            
            loss_t = criterion(out_t, label)*args.alpha
            loss_t.backward()

            gs_plugin.before_update(model.fusion_model.fc_out, t, 
                                    batch_step, len_dataloader, gs_plugin.exp_count)
            optimizer.step()
            optimizer.zero_grad()

            gs_plugin.exp_count += 1

        for n, p in model.named_parameters():
            if p.grad != None:
                del p.grad

        _loss += (loss_a * av_alpha + loss_v * (1 - av_alpha)).item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        if args.modal3:
            _loss_t += loss_t.item()

        # 更新 tqdm 后缀信息（平均损失）
        try:
            data_iter.set_postfix({
                'loss': _loss / (batch_step + 1),
                'loss_a': _loss_a / (batch_step + 1),
                'loss_v': _loss_v / (batch_step + 1)
            })
        except Exception:
            pass
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"  Epoch Duration: {epoch_duration:.2f} seconds ({epoch_duration/60:.2f} minutes)")
        
    scheduler.step()
    if args.modal3:
        return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _loss_t / len(dataloader), epoch_duration
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), epoch_duration

def valid(args, model, device, dataloader, 
          gs_flag = False, av_alpha = 0.5, 
          a_alpha = 0.35, v_alpha = 0.25, t_alpha = 0.4):
    softmax = nn.Softmax(dim=1)
    modal_names = ast.literal_eval(args.model_name)
    n_classes = args.num_classes
    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        pred_result = []
        data_iter = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid", ncols=120)
        for step, data_packet in data_iter:
                # 加载数据
                if modal_names == ["Visual", "Audio"]:
                    spec, image,label  = data_packet[0],data_packet[1],data_packet[2]
                    # spec, image, label, _ = batch
                    spec, image, label = spec.to(device), image.to(device), label.to(device)
                    if args.dataset == 'CREMAD':
                        data_mini_packet = (image.float(), spec.float())
                    else:
                        data_mini_packet = (image.float(), spec.unsqueeze(1).float())
                elif modal_names == ["Image", "Text"]:
                    token, padding_mask, image, label, _ = data_packet
                    token, padding_mask = token.to(device), padding_mask.to(device)
                    image, label = image.to(device), label.to(device)
                    data_mini_packet = (token, padding_mask, image)
                elif modal_names == ["Text", "Visual", "Audio"]:
                    token, padding_mask, image, spec, label, _ = data_packet
                    token, padding_mask = token.to(device), padding_mask.to(device)
                    image, spec, label = image.to(device), spec.to(device), label.to(device)
                    data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
                 # 对第一个模态进行GS训练
                if len(modal_names) == 2:
                    v,a = model(data_mini_packet)
                else:
                    t,v,a = model(data_mini_packet)
                # out_a = model.fusion_model.fc_out(a)
                # out_v = model.fusion_model.fc_out(v)
                out_a = model.fusion_model.fc_out(a)
                out_v = model.fusion_model.fc_out(v)
                if len(modal_names) == 3:
                    out_t = model.fusion_model.fc_out(t)
                if args.dynamic:
                    if args.modal3:
                        w_map, e_map = calculate_gating_weights({'audio': out_a, 'visual': out_v, 'text': out_t})
                        out = out_a * w_map['audio'] + out_v * w_map['visual'] + out_t * w_map['text']
                    else:
                        w_map, e_map = calculate_gating_weights({'audio': out_a, 'visual': out_v})
                        out = out_a * w_map['audio'] + out_v * w_map['visual']
                    # if args.modal3:
                    #     audio_conf, img_conf, txt_conf = calculate_gating_weights3(out_a, out_v, out_t)
                    #     out = (out_a * audio_conf + out_v * img_conf + out_t * txt_conf)
                    # else:
                    #     txt_conf, img_conf = calculate_gating_weights(out_a, out_v)
                    #     out = (out_a * txt_conf + out_v * img_conf)
                else:
                    if args.modal3:
                        out = a_alpha * out_a + v_alpha * out_v + t_alpha * out_t
                    else:
                        out = av_alpha * out_a + (1-av_alpha) * out_v
    
                prediction = softmax(out)
                pred_v = softmax(out_v)
                pred_a = softmax(out_a)
                if args.modal3:
                    pred_t = softmax(out_t)
    
                for i in range(image.shape[0]):
    
                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    v = np.argmax(pred_v[i].cpu().data.numpy())
                    a = np.argmax(pred_a[i].cpu().data.numpy())
                    if args.modal3:
                        t = np.argmax(pred_t[i].cpu().data.numpy())
                    num[label[i]] += 1.0
    
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0
                    if args.modal3:
                        if np.asarray(label[i].cpu()) == t:
                            acc_t[label[i]] += 1.0
    
                # 更新 tqdm 后缀信息（running accuracy）
                try:
                    processed = sum(num)
                    running_acc = sum(acc) / processed if processed > 0 else 0.0
                    data_iter.set_postfix({'acc': running_acc})
                except Exception:
                    pass
    if args.modal3:
        return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_t) / sum(num)    
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir):
    all_ckpts = os.listdir(exp_dir)
    sdA = torch.load(os.path.join(exp_dir, all_ckpts[0]), map_location='cpu')["model"]
    model_cnt = 1
    for epoch in range(1, len(all_ckpts)):
        sdB = torch.load(os.path.join(exp_dir, all_ckpts[epoch]), map_location='cpu')["model"]
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, 1, len(all_ckpts)))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA


def main(av_alpha = 0.5):
    args = get_arguments()
    # print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    # ==================数据集加载=====================================
    if args.dataset == 'CREMAD':
        args.num_classes = 6
        # train_dataset = CramedDataset(mode='train', args=args)
        # test_dataset = CramedDataset(mode='test', args=args)
        train_dataset = AV_CD_Dataset(mode='train')
        test_dataset = AV_CD_Dataset(mode='test')
        print(f"use train AV_CD_Dataset and test AV_CD_Dataset for CREMAD dataset")
    elif args.dataset == 'KineticSound':
        args.num_classes = 34
        train_dataset = KSDataset(mode='train', args=args)
        test_dataset = KSDataset(mode='test', args=args)
    elif args.dataset == 'AVE':
        args.num_classes = 28 
        train_dataset = AVEDataset(mode='train', args=args)
        test_dataset = AVEDataset(mode='test', args=args)
    elif args.dataset == 'Food101':
        args.num_classes = 101
        train_dataset = M3AEDataset(args,mode='train')
        test_dataset = M3AEDataset(args,mode='test')
    elif args.dataset == 'MVSA':
        args.num_classes = 3
        train_dataset = M3AEDataset(args,mode='train')
        test_dataset = M3AEDataset(args,mode='test')
    elif args.dataset == 'IEMOCAP3':
        args.num_classes = 5
        train_dataset = TVADataset(mode='train', args=args, pick_num=3)
        test_dataset = TVADataset(mode='test', args=args, pick_num=3)
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support AVE, KineticSound and CREMA-D for now!'.format(args.dataset))
    print("train_dataset size: {}".format(len(train_dataset)))
    print("test_dataset size: {}".format(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    if args.model_name == '["Visual","Audio"]' or args.model_name == '["Image","Text"]':
        print(f"Using {args.model_name} model for training.")
        model = VA_MLA_Classifier(args)
    else:
        model = TVA_MLA_Classifier(args)
    if args.ckpt_load_path_train:
        loaded_dict = torch.load(args.ckpt_load_path_train)
        state_dict = loaded_dict['model']
        state_dict = {key[7:]: state_dict[key] for key in state_dict}
        del state_dict["fusion_module.fc_out.weight"]
        del state_dict["fusion_module.fc_out.bias"]
        missing, unexcepted = model.load_state_dict(state_dict, strict = False)
        print('Trained model loaded!')
    
    model.to(device)
    print_model_params(model)
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()
    if args.model_name == '["Visual","Audio"]' and args.Use_initWeight:
        model.apply(weight_init)
        print(f"Use Weight int")

    if args.lorb == "large" and args.cav_opti:
        # optimizer = optim.SGD(model.fusion_model.fc_out.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        mlp_list = ['fusion_module.fc_out.weight', 'module.fusion_module.fc_out.bias']
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]
        optimizer = optim.Adam([{'params': base_params, 'lr': args.learning_rate / 10}, 
                                {'params': mlp_params, 'lr': args.learning_rate}],
                                weight_decay=5e-7, 
                                betas=(0.95, 0.999))
    else:
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = 1e-4, betas=(0.9, 0.999))
    if args.lorb == "large" and args.cav_lrs:
        args.lrscheduler_start = 2
        args.lrscheduler_step = 1
        args.lrscheduler_decay = 0.5
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
                                                    gamma = args.lrscheduler_decay)    

    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

   
    # GS Plugin
    gs = GSPlugin()
    txt_history = None
    img_history = None
    audio_history = None    
    print(f"args num_class is {args.num_classes}")
    if args.train:

        best_acc = 0.0
        if args.gs_flag:
            log_name = '{}_{}_{}'.format(args.fusion_method, "GS", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            log_name = '{}_{}_{}'.format(args.fusion_method, args.modulation, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        epoch_times = []
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset, log_name)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                writer = SummaryWriter(writer_path)

                if args.modal3:
                    batch_loss, batch_loss_a, batch_loss_v, batch_loss_t, epoch_duration = train_epoch(args, epoch, model, device, 
                                                                     train_dataloader, optimizer,
                                                                     scheduler, gs_plugin = gs, 
                                                                     writer = writer, 
                                                                     gs_flag = args.gs_flag, 
                                                                     av_alpha = av_alpha,
                                                                     txt_history = txt_history,
                                                                     img_history = img_history,
                                                                     audio_history=audio_history)
                    
                    acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
                                            av_alpha= av_alpha, 
                                            gs_flag= args.gs_flag,
                                            a_alpha= args.a_alpha,
                                            v_alpha= args.v_alpha,
                                            t_alpha= args.t_alpha)

                    writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                                'Audio Loss': batch_loss_a,
                                                'Visual Loss': batch_loss_v,
                                                'Text Loss': batch_loss_t}, epoch)

                    writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                    'Audio Accuracy': acc_a,
                                                    'Visual Accuracy': acc_v,
                                                    'Text Accuracy': acc_t}, epoch)
                else:
                    print(gs)
                    batch_loss, batch_loss_a, batch_loss_v, epoch_duration = train_epoch(args, epoch, model, device, 
                                                                        train_dataloader, optimizer,
                                                                        scheduler, gs_plugin = gs, 
                                                                        writer = writer, 
                                                                        gs_flag = args.gs_flag, 
                                                                        av_alpha = av_alpha,
                                                                        txt_history = txt_history,
                                                                        img_history = img_history)
                    acc, acc_a, acc_v = valid(args, model, device, test_dataloader, 
                                            av_alpha= av_alpha, 
                                            gs_flag= args.gs_flag)

                    writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                                'Audio Loss': batch_loss_a,
                                                'Visual Loss': batch_loss_v}, epoch)

                    writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                    'Audio Accuracy': acc_a,
                                                    'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v, epoch_duration = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
            epoch_times.append(epoch_duration)
            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                if args.modal3:
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
                else:    
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                if args.modal3:
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f}, Text Acc: {:.3f} ".format(acc_a, acc_v, acc_t))
                else:    
                    print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
            # 计算并输出训练时长统计
            if epoch_times:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                total_train_time = sum(epoch_times)
                print(f"\n=== Training Time Statistics ===")
                print(f"  Average Epoch Time: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)")
                print(f"  Total Training Time: {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes, {total_train_time/3600:.2f} hours)")

    else:
        # if args.lorb == "large":
        #     state_dict = wa_model("ckpt/")
        # else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']

        missing, unexcepted = model.load_state_dict(state_dict)
        print('Trained model loaded!')
        
        if not args.modal3:
            acc, acc_a, acc_v = valid(args, model, device, 
                                      test_dataloader, args.ewc_flag, args.gs_flag, args.av_alpha)
            print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
        else:
            acc, acc_a, acc_v, acc_t = valid(args, model, device, test_dataloader, 
                                             args.ewc_flag, args.gs_flag, args.av_alpha,
                                             a_alpha= args.a_alpha, v_alpha= args.v_alpha, t_alpha= args.t_alpha)
            print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}, accuracy_t: {}'.format(acc, acc_a, acc_v, acc_t))


if __name__ == "__main__":
    main(av_alpha = 0.55)
