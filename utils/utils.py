import torch
import random
import numpy as np

import torch.nn as nn
from torch.nn import functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def print_model_params(model):
    total_trainable = 0
    print("-" * 100)
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_trainable += num_params
            # print(f"{name:60} | 可训练: {'是':8} | 形状: {str(list(param.shape)):20} | 参数数量: {num_params:,}")
    print("-" * 100)
    print(f"总可训练参数数量: {total_trainable:,}")
    print(f"模型总参数数量: {sum(p.numel() for p in model.parameters()):,}")



def get_optimizer_lr(opt):
    if opt is None:
        return None
    try:
        lrs = [g['lr'] for g in opt.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs
    except Exception:
        return None

def print_current_lrs(opt_m1, opt_m2, opt_fusion, opt_m3, sched_map=None):
    parts = []
    if opt_m1 is not None:
        parts.append(f"LR_m1={get_optimizer_lr(opt_m1)}")
    if opt_m2 is not None:
        parts.append(f"LR_m2={get_optimizer_lr(opt_m2)}")
    if opt_m3 is not None:
        parts.append(f"LR_m3={get_optimizer_lr(opt_m3)}")
    if opt_fusion is not None:
        parts.append(f"LR_fusion={get_optimizer_lr(opt_fusion)}")
    # 也尝试打印 scheduler 的 last lr（若可用）
    if sched_map:
        try:
            sch_lrs = []
            for sch in sched_map:
                if hasattr(sch, 'get_last_lr'):
                    sch_lrs.append(sch.get_last_lr())
                else:
                    sch_lrs.append(None)
            parts.append(f"Scheduler_last_lrs={sch_lrs}")
        except Exception:
            pass
    print("Current learning rates: " + ", ".join(map(str, parts)))

# LFM的需要的Functions
def Alignment(p, q):
    """ Compute the Jensen-Shannon Divergence between two probability distributions. """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * p + 0.5 * q
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def getAlpha_Learnable_Fitted(epoch):
    # Alpha with Learnable learning is fitted with functions
    coef_alpha1 = [2.04623704e-01, 3.35472727e-03, 1.22989557e-04, -2.92947416e-06, 2.23835486e-08, -5.39717505e-11]
    coef_alpha2 = [7.95376296e-01, -3.35472727e-03, -1.22989557e-04, 2.92947416e-06, -2.23835486e-08, 5.39717505e-11]
    alpha1 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha1))
    alpha2 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha2))
    return [alpha1, alpha2]



