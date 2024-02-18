import torch
from torchmetrics import F1Score, Accuracy
from nltk.metrics import edit_distance

def calc_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.detach().cpu().numpy()/max_indices.size()[0]
    return train_acc


def accuracy(num_classes, pred, target, device="cuda"):
    return Accuracy(num_classes=num_classes).to(device)(pred.view(-1,), target.view(-1,))


def f1_score(num_classes, pred, target, device="cuda"):
    return F1Score(num_classes=num_classes).to(device)(pred, target)


def WER(predicted, actual):
    total_errors = 0
    total_words = 0

    for pred_seq, actual_seq in zip(predicted, actual):
        # 단어 단위 시퀀스로 가정
        pred_seq_list = pred_seq  # 또는 필요에 따라 변환
        actual_seq_list = actual_seq  # 또는 필요에 따라 변환

        edit_dist = edit_distance(pred_seq_list, actual_seq_list)
        total_errors += edit_dist
        total_words += len(actual_seq_list)  # 참조 텍스트의 단어 수를 기준으로

    wer = float(total_errors) / float(total_words) * 100
    return wer