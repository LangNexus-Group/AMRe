import torch
from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(logits, labels):
    """
    从模型的 logits 输出和真实标签计算准确率和 F1 分数。

    Args:
        logits (torch.Tensor): 模型的原始输出，形状为 [batch_size, num_classes] 或 [batch_size, 1]。
        labels (torch.Tensor): 真实标签，形状为 [batch_size]。

    Returns:
        tuple: (accuracy, f1)
    """
    # 将 logits 转换为预测类别
    if logits.dim() > 1 and logits.shape[1] > 1:
        # 多分类情况 (CrossEntropyLoss)
        preds = torch.argmax(logits, dim=1)
    else:
        # 二分类情况 (BCEWithLogitsLoss)
        preds = (torch.sigmoid(logits) > 0.5).squeeze()

    # 将 Tensor 移动到 CPU 并转换为 numpy 数组
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算指标
    acc = accuracy_score(labels_np, preds_np)
    # 对于二分类，average='binary'；对于多分类，可使用 'macro' 或 'weighted'
    f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0) 
    
    return acc, f1