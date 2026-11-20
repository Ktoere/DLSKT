# @File    : train.py
# @Software: PyCharm


import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
import yaml
from dataloader import Data_set
from model.DLSKT import DLSKTnet
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import metrics





def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    all_pred = np.array(all_pred)
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred = np.array(all_pred)
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

parser = argparse.ArgumentParser(description='myDemo')
parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: value in dlsktconf.yml)')
parser.add_argument('--data_dir', type=str, default='dataset/Processed_data/')
parser.add_argument('--patience', type=int, default=5,
                    help='early-stopping patience based on validation loss')
parser.add_argument('--min_delta', type=float, default=0.0,
                    help='minimum validation-loss decrease counted as an improvement')
parser.add_argument('--split_seed', type=int, default=42,
                    help='random seed used to split validation and test data')

# Press the green button in the gutter to run the script.


def train(datasetname,model,train_dataloader,optimizer, criterion,device):
    model.train()
    train_loss = []
    label_num = []
    outs = []
    for exercise_seq, concept_seq, response_seq, attemptCount_seq, hintCount_seq, taken_time_seq, interval_time_seq in tqdm(
            train_dataloader,
            desc='Training',
            mininterval=2):
        exercise_seq = exercise_seq.long().to(device)
        concept_seq = concept_seq.long().to(device)
        response_seq = response_seq.long().to(device)
        attemptCount_seq = attemptCount_seq.long().to(device)
        hintCount_seq = hintCount_seq.long().to(device)

        taken_time_seq = taken_time_seq.long().to(device)
        interval_time_seq = interval_time_seq.long().to(device)
        mask_ex = exercise_seq[:, 1:].clone()

        target = response_seq[:, 1:].float().clone()

        optimizer.zero_grad()
        output, distillation_loss, mi_loss = model(exercise_seq, concept_seq, response_seq, attemptCount_seq,
                                                   hintCount_seq, taken_time_seq, interval_time_seq)
        mask = mask_ex > 0
        masked_pred = output[mask]
        masked_truth = target[mask]

        cross_loss = criterion(masked_pred, masked_truth)
        # total = cross_loss + distillation_loss + mi_loss
        # a = cross_loss/total
        # b = distillation_loss/total
        # c = mi_loss/total

        final_loss = cross_loss + distillation_loss * 0.8 + mi_loss * 0.8
        final_loss.backward()
        optimizer.step()
        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        train_loss.append(final_loss.item())

        label_num.extend(masked_truth)
        outs.extend(masked_pred)

    loss = np.average(train_loss)

    # loss = binaryEntropy(label_num, outs)
    auc = compute_auc(label_num, outs)
    accuracy = compute_accuracy(label_num, outs)
    rmse_sklearn = np.sqrt(mean_squared_error(label_num, outs,))
    return loss, accuracy, auc,rmse_sklearn



def test_epoch(datasetname,model,test_loader,criterion,device="cpu"):
    model.eval()
    train_loss = []

    label_num = []
    outs = []
    for exercise_seq, concept_seq, response_seq, attemptCount_seq, hintCount_seq, taken_time_seq, interval_time_seq in tqdm(
            test_loader, desc='Testing',
            mininterval=2):
        exercise_seq = exercise_seq.to(device)
        concept_seq = concept_seq.to(device)
        response_seq = response_seq.to(device)
        attemptCount_seq = attemptCount_seq.long().to(device)
        hintCount_seq = hintCount_seq.long().to(device)
        taken_time_seq = taken_time_seq.to(device)
        interval_time_seq = interval_time_seq.to(device)
        mask_ex = exercise_seq[:, 1:].clone()

        target = response_seq[:, 1:].float().clone()
        with torch.no_grad():
            output, distillation_loss, mi_loss = model(exercise_seq, concept_seq, response_seq, attemptCount_seq,
                                                       hintCount_seq, taken_time_seq, interval_time_seq)

        mask = mask_ex > 0
        masked_pred = output[mask]
        masked_truth = target[mask]
        cross_loss = criterion(masked_pred, masked_truth)
        final_loss = cross_loss + distillation_loss * 0.8 + mi_loss * 0.8
        # final_loss = cross_loss
        # final_loss = a * cross_loss + b * distillation_loss + c * mi_loss

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        train_loss.append(final_loss.item())
        label_num.extend(masked_truth)
        outs.extend(masked_pred)







    loss = np.average(train_loss)
    # loss = binaryEntropy(label_num, outs)
    auc = compute_auc(label_num,outs)
    acc = compute_accuracy(label_num, outs)
    rmse_sklearn = np.sqrt(mean_squared_error(label_num, outs))
    return loss, acc, auc,rmse_sklearn


if __name__ == "__main__":

    datasetname = "ASSIST17"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if datasetname == 'ASSIST09':
        parser.add_argument('--datasetname', type=str, default='ASSIST09', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train NIPS34 file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train NIPS34 file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2009/',
                            help="train NIPS34 file, default as './ASSISTments2009/'.")
        # parser.add_argument('--n_question', type=int, default=123, help='the number of unique questions in the dataset')

    elif datasetname == 'ASSIST12':
        parser.add_argument('--datasetname', type=str, default='ASSIST12', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train ASSIST12 file, default as 'train_data.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train ASSIST12 file, default as 'test_data.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSIST2012/',
                            help="train ASSIST12 file, default as './ASSIST2012/'.")
        # parser.add_argument('--n_question', type=int, default=53091, help='the number of unique questions in the dataset')


    elif datasetname == 'ASSIST17':
        parser.add_argument('--datasetname', type=str, default='ASSIST17', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train ASSIST17 file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train ASSIST17 file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSIST2017/',
                            help="train ASSIST17 file, default as './ASSIST2017/'.")
        # parser.add_argument('--n_question', type=int, default=3162, help='the number of unique questions in the dataset')


    elif datasetname == 'JUNYI':
        parser.add_argument('--datasetname', type=str, default='JUNYI', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='/train_data.csv',
                            help="train JUNYI file, default as 'train_data.csv'.")
        parser.add_argument('--test_file', type=str, default='/test_data.csv',
                            help="train JUNYI file, default as 'test_data.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./JUNYI/',
                            help="train JUNYI file, default as './JUNYI/'.")
        # parser.add_argument('--n_question', type=int, default=1326, help='the number of unique questions in the dataset')
        # parser.add_argument('--n_question', type=int, default=25784,
        #                     help='the number of unique questions in the dataset')



    parsers = parser.parse_args()
    datasetname = parsers.datasetname
    print("parser:", parsers)
    print(f'loading Dataset  {parsers.datasetname}...')


    with open("dlsktconf.yml", 'r', encoding='utf-8') as f:
        tnktconf = yaml.safe_load(f)
    with open("dataset.yml", 'r', encoding='utf-8') as f:
        dataset_cof = yaml.safe_load(f)
    input_dim = tnktconf["train"]["input_dim"]
    student_num = dataset_cof[datasetname.lower()]["student_number"]
    exercise_size = dataset_cof[datasetname.lower()]["exercise_size"]
    concept_size = dataset_cof[datasetname.lower()]["concept_size"]
    hidden_dim = tnktconf["train"]["hidden_dim"]
    seq_max_length = tnktconf["train"]["max_seq_length"]

    seed = tnktconf["train"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_path = parsers.data_dir + parsers.datasetname + parsers.train_file
    test_path = parsers.data_dir + parsers.datasetname + parsers.test_file
    train_set = Data_set(path=train_path, max_seq_length=seq_max_length)
    test_set = Data_set(path=test_path, max_seq_length=seq_max_length)
    train_loader = DataLoader(train_set, tnktconf["train"]["batch_size"], shuffle=True)
    total_len = len(test_set)
    val_len =  total_len // 3
    test_len = total_len - val_len
    generator = torch.Generator().manual_seed(parsers.split_seed)
    val_set, new_test_set = random_split(test_set, [val_len, test_len], generator=generator)
    val_loader = DataLoader(val_set, batch_size=tnktconf["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(new_test_set, batch_size=tnktconf["train"]["batch_size"], shuffle=False)
    print(f"dataset split: train={len(train_set)}, validation={val_len}, test={test_len}")

    # train_set = Data_set(path=train_path, max_seq_length=seq_max_length)
    # test_set = Data_set(path=val_path, max_seq_length=seq_max_length)
    # test_set = Data_set(path=test_path, max_seq_length=seq_max_length)
    # train_loader = DataLoader(train_set, tnktconf["train"]["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=tnktconf["train"]["batch_size"])
    # test_loader = DataLoader(test_set, batch_size=tnktconf["train"]["batch_size"])
    kt_model = DLSKTnet(exercise_size, concept_size, input_dim,dataset_cof[datasetname.lower()]["timeTaken"],dataset_cof[datasetname.lower()]["interval_time"],tnktconf["train"],dataset_cof[datasetname.lower()])



    if tnktconf["optimizer"]["name"] == "adam":
        opt = torch.optim.Adam(kt_model.parameters(), lr=tnktconf["optimizer"]["lr"], betas=(0.9, 0.9999), eps=1e-8, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)


    criterion = nn.BCELoss(reduction="mean")
    kt_model.to(device)
    criterion.to(device)

    patience = parsers.patience
    if patience < 1:
        raise ValueError("--patience must be at least 1")

    max_epochs = parsers.epochs if parsers.epochs is not None else tnktconf["train"]["epoch"]
    if max_epochs < 1:
        raise ValueError("--epochs must be at least 1")

    checkpoint_dir = Path(parsers.save_dir_prefix)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"DLSKT_best_{datasetname}.pth"

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = -1
    best_val_metrics = None

    for epoch in range(max_epochs):
        train_loss, train_acc, train_auc,train_rmse = train(datasetname,kt_model, train_loader,opt,criterion,device)
        scheduler.step()

        print(
            "epoch - {} train_loss - {:.2f} acc - {:.4f} auc - {:.4f}  rmse - {:.4f}".format(epoch, train_loss, train_acc, train_auc,train_rmse))

        val_loss, val_acc, val_auc,val_rmse = test_epoch(datasetname,kt_model,val_loader, criterion, device=device)
        print("epoch - {} val_loss - {:.2f} acc - {:.4f} auc - {:.4f}  rmse - {:.4f}".format(epoch, val_loss, val_acc, val_auc,val_rmse))


        if np.isfinite(val_loss) and val_loss < best_val_loss - parsers.min_delta:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_val_metrics = {
                "loss": float(val_loss),
                "acc": float(val_acc),
                "auc": float(val_auc),
                "rmse": float(val_rmse),
            }
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": kt_model.state_dict(),
                    "val_metrics": best_val_metrics,
                    "dataset": datasetname,
                },
                checkpoint_path,
            )
            print(f"validation loss improved; checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"validation loss did not improve ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"early stopping triggered at epoch {epoch}; best epoch was {best_epoch}")
            break

    if best_epoch < 0:
        raise RuntimeError("training finished without a finite validation loss; no checkpoint was saved")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    kt_model.load_state_dict(checkpoint["model_state_dict"])
    print(
        "restored best checkpoint: epoch - {} val_loss - {:.4f} acc - {:.4f} "
        "auc - {:.4f} rmse - {:.4f}".format(
            best_epoch,
            best_val_metrics["loss"],
            best_val_metrics["acc"],
            best_val_metrics["auc"],
            best_val_metrics["rmse"],
        )
    )

    test_loss, test_acc, test_auc, test_rmse = test_epoch(
        datasetname, kt_model, test_loader, criterion, device=device
    )
    print(
        "final test: loss - {:.4f} acc - {:.4f} auc - {:.4f} rmse - {:.4f}".format(
            test_loss, test_acc, test_auc, test_rmse
        )
    )

