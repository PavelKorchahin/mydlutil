# Evaluation metrics for classification tasks and models for classification tasks
# The shape of the output tensor of model net is [batch size, number of categories, ...]
# In the dimension of the number of categories, the probabilities of each category are stored
# The shape of the real label tensor is [batch size, ...] Compared with the output tensor,
# the number of categories in this dimension is reduced, while the other dimensions are the same

# The classfication evaluation metrics include:
# confusion matrix, accuracy, precision, recall, F1 score, miou, fwmiou, roc curve, auc, etc.

# The model evaluation metrics include:
# calculating the effective receptive field;
# Represent the changes in evaluation metrics of different datasets on different networks;
# Calculate the number of parameters and flops，etc.
from typing import Literal, Union, Sequence
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import torch
from matplotlib import pyplot as plt
from ptflops import get_model_complexity_info
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from . import img, charts as charts
from .load_data import GenericDataset


class ClassificationEvaluate:
    """
    The metrics for a classification model.
    Args:
        net (nn.Module): The network to be evaluated.
        test_iter (DataLoader[GenericDataset]): The dataloader of the dataset the model ``net`` based on which must be an obeject of ``torch.utils.data.Dataset`` .
        num_class (int): The number of classes.
        class_label (list[str], optinal): The labels of the classes.
            If not specified, the attribute will be obtained from the attribute of ``test_iter.dataset``,
            in which case ``test_iter.dataset`` must be an object of ``GenericDataset``.
            See ``DatasetConfig.class_label`` for more information.
        pixelwise (bool, optional): whether the labels are pixelwised about features.
            If not specified, the attribute will be obtained from the attribute of ``test_iter.dataset``,
            in which case ``test_iter.dataset`` must be an object of ``GenericDataset``.
        device (str): The device to use.
            Default is ``'cuda:0'``.
        roc_curve (bool): Whether to calculate the ROC curve.
            It's easy to be out of memory when caculating a ROC curve when the test data is large.
            One can sepecfy param ``batch_per_roc_sample_point`` to reduce the computational quantity and memory usage.
            Default is False.
        batch_per_roc_sample_point (int): The number of batches used to calculate a single point on the roc curve.
            The larger the test data, the larger this parameter should be specified to avoid out of memory.
            Default is 1.
        mode ('train', 'test', 'img', optional): The mode of the dataset which is the model ``net`` based on.
            Default is 'test'.
            See ``DatasetConfig.mode`` for more information.
    """
    def __init__(
            self,
            net: nn.Module,
            test_iter: DataLoader[Dataset],
            num_class: int,
            class_label: list[str] = None,
            pixelwise: bool = None,
            device: str = 'cuda:0',
            roc_curve: bool = False,
            batch_per_roc_sample_point: int = 1,
            mode: Literal['train', 'test', 'img'] = 'test'
    ):

        self.net = net.to(device)
        net = self.net
        self.device = device
        self.num_class = num_class
        cm = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)
        pred_num = torch.zeros(num_class, dtype=torch.int64, device=device)
        true_num = torch.zeros(num_class, dtype=torch.int64, device=device)
        all_probs = []
        all_labels = []
        size = 0
        if mode == 'test':
            net.eval()
        else:
            net.train()
        self.mode = mode
        with torch.no_grad():
            k = 0
            it = tqdm(test_iter, leave=False, desc=f'evaluating {mode} dataset: ', )
            for data, label in it:
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                output = net(data)
                if output.dim() == 4:
                    output = output.permute(0, 2, 3, 1).reshape(-1, self.num_class)
                    label = label.reshape(-1)
                valid_mask = label < num_class
                output = output[valid_mask]
                label = label[valid_mask]
                pred = output.argmax(dim=1)
                if label.numel() == 0:
                    continue
                size += label.numel()
                idx = label * num_class + pred
                cm += torch.bincount(idx, minlength=num_class * num_class).reshape(num_class, num_class)
                pred_num += torch.bincount(pred, minlength=num_class)
                true_num += torch.bincount(label, minlength=num_class)
                k += 1
                if roc_curve and k % batch_per_roc_sample_point == 0:
                    all_probs.append(output.detach().cpu())
                    all_labels.append(label.detach().cpu())
        it.refresh()
        it.close()
        if roc_curve:
            all_probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, num_class))
            all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0, ), dtype=torch.long)
        self.confusion_matrix = cm.int().tolist()
        self.true_num = true_num.int().tolist()
        self.total_true = int(cm.diag().sum().item())
        self.pred_num = pred_num.int().tolist()
        self.all_probs = all_probs
        self.all_labels = all_labels
        self.size = size
        if class_label is None:
            class_label = getattr(test_iter.dataset, 'class_label')
        if pixelwise is None:
            pixelwise = getattr(test_iter.dataset, 'pixelwise')
        self.dataset = test_iter.dataset
        self.class_label = class_label
        self.pixelwise = pixelwise
        self.roc_curve = roc_curve

    def get_result(
            self,
            start_idx: int = 0,
            num: int = 10
    ) -> list[tuple]:
        """

        Get the result of model's output.
        Args:
            start_idx (int, optional): The starting index of samples in the dataset the model ``net`` based on.
                Default is 0.
            num (int, optional): The number of samples to be returned.
                Default is 10.

        Returns:
            list[tuple]: A list of tuples containing the sample, predicted label, and ground truth label.
                The length of list is equal to ``num``
        """
        result = []
        for idx in range(start_idx, start_idx+num):
            data, label = self.dataset[idx]
            out = self.net(data.unsqueeze(0).to(self.device)).argmax(dim=1).squeeze(0)
            result.append((data, out, label))
        return result

    def p(self, k: int) -> int:
        """
        get the number of samples whose true class number is k.
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The number of samples whose true class number is k.
        """
        return self.true_num[k]

    def n(self, k: int) -> int:
        """
        Get the number of samples whose true class number is not k.
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The number of samples whose true class number is not k.

        """
        return self.size - self.true_num[k]

    def t(self, k: int | None = None) -> int:
        """
        Get the number of samples whose predicted correctly.
        Args:
            k (None | int, optional):
                - None (default): It will get the total number of samples whose predicted correctly.

                - int: it will get the number of samples
                    whose true class number is k and predicted class number is also k
                    or true class number is not k and predicted class number is also not k
                    It is focus on the prediction correctness whether it is class k or not,
                    regardless of actual prediction correctness
                    whether the predicted class number and true class number are the same or not.
                    For example, if a sample whose true class number is i and predicted class number is j,
                    it will be counted if i == j == k or i != k and j != k whether i == j or not.
        Returns:
            The number of samples whose predicted correctly if ``k`` is None;
            The number of samples whose true class number is k and predicted class number is also k
            or true class number is not k and predicted class number is also not k
            if ``k`` is a int.

        """
        if k is not None:
            return self.size - self.true_num[k] - self.pred_num[k] + 2 * self.confusion_matrix[k][k]
        return self.total_true

    def f(self, k: int | None = None) -> int:
        """
        Get the number of samples whose predicted incorrectly.
        Args:
            k (None | int, optional):
                - None (default): It will get the total number of samples whose predicted incorrectly.

                - int: it will get the number of samples whose true class number is k but predicted class number is not k
                  or true class number is not k but predicted class number is k.

        Returns:
            The number of samples whose predicted incorrectly if ``k`` is None;
            The number of samples whose whose true class number is k but predicted class number is not k
            or true class number is not k but predicted class number is k.
            if ``k`` is a int.

        """
        if k is not None:
            return self.true_num[k] + self.pred_num[k] - self.confusion_matrix[k][k]
        return self.size - self.total_true

    def tp(self, k: int) -> float:
        """
        Get the number of samples whose true class number is k and predicted class number is also k, namely the TP samples for class k.
        Args:
            k:  k (int): The class number to be analyzed.

        Returns:
            the number of samples whose true class number is k and predicted class number is also k,
            namely the TP samples for class k.
        """
        return self.confusion_matrix[k][k]

    def fp(self, k: int) -> float:
        """
        Get the number of samples whose true class number is not k but predicted class number is k, namely the FP samples for class k.
        Args:
            k:  k (int): The class number to be analyzed.

        Returns:
            the number of samples whose true class number is not k but predicted class number is k,
            namely the FP samples for class k.
        """
        return self.pred_num[k] - self.confusion_matrix[k][k]

    def tn(self, k: int) -> float:
        """
        Get the number of samples whose true class number is not k and predicted class number is also not k, namely the TN samples for class k.
        It is focus on the prediction correctness whether it is not class k,
        regardless of actual prediction correctness.
        For example, if a sample whose true class number is i and predicted class number is j,
        it will be counted if i != k and j != k whether i == j or not
        Args:
            k (int): The class number to be analyzed.

        Returns:
            the number of samples whose true class number is k and predicted class number is also k,
            namely the TN samples for class k.
        """
        return self.size - self.true_num[k] - self.pred_num[k] + self.confusion_matrix[k][k]

    def fn(self, k: int) -> float:
        """
        Get the number of samples whose true class number is not k and predicted class number is also not k, namely the FN samples for class k.
        Args:
            k (int): The class number to be analyzed.

        Returns:
            the number of samples whose true class number is not k and predicted class number is also not k,
            namely the FN samples for class k.
        """
        return self.true_num[k] - self.confusion_matrix[k][k]

    def cm(self, k: int | None = None) -> list[list[int]]:
        """
        Get confusion matrix.
        Args:
            k (None | int, optional):
            - None (default): It will get overall confusion matrix.

            - int: it will get the confusion matrix of class k
              which is equivalent to a binary confusion matrix,
              with one class number being k and the other not being k

        Returns:
            The overall confusion matrix if ``k`` is None;
            The confusion matrix of class k if ``k`` is a int.

        """
        if k is not None:
            return [[self.tp(k), self.fn(k)], [self.fp(k), self.tn(k)]]
        return self.confusion_matrix

    def precision(
            self,
            k: int,
            zero_define: int = 0
    ) -> float:
        """
        Get the precision of class k, namely TP(k) / (TP(k) + FP(k)).
        Args:
            k (int): The class number to be analyzed.
            zero_define (int, optional): The value defined when None of the prediction results are class k, namely TP(k) + FP(k) =0.
                Default is 0.
        Returns:
            The precision of class k.
        """

        prediction_positive_num = self.tp(k) + self.fp(k)

        return self.tp(k) / prediction_positive_num if prediction_positive_num != 0 else zero_define

    def npv(
            self,
            k: int,
            zero_define: int = 0
    ) -> float:
        """
        Get the npv (negative predictive value) of class k, namely TN(k) / (TN(k) + FN(k)).
        Args:
            k (int): The class number to be analyzed.
            zero_define (int, optional): The value defined when All of the prediction results are class k, namely TN(k) + FN(k) =0.
                Default is 0.
        Returns:
            The npv (negative predictive value) of class k.
        """
        prediction_negative_num = self.tn(k) + self.fn(k)
        return self.tn(k) / prediction_negative_num if prediction_negative_num != 0 else zero_define

    def recall(self, k: int) -> float :
        """
        Get the recall of class k, namely TP(k) / (TP(k) + FN(k)).
        It is allowed to be called by its alias ``sensitivity`` or ``tpr`` .
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The recall of class k.
        """
        return self.tp(k) / (self.tp(k) + self.fn(k))

    def specificity(self, k: int) -> float:
        """
        Get the specificity of class k, namely TN(k) / (TN(k) + FN(k)).
        It is allowed to be called by its alias ``tnr`` .
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The specificity of class k.
        """
        return self.tn(k) / (self.tn(k) + self.fp(k))

    def fpr(self, k: int) -> float:
        """
        Get the fpr(false positive rate) of class k, namely FP(k) / (TN(k) + FP(k)).
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The fpr(false positive rate) of class k.
        """
        return self.fp(k) / (self.fp(k) + self.tn(k))

    def fnr(self, k: int) -> float:
        """
        Get the fnr(false negative rate) of class k, namely FN(k) / (TP(k) + FN(k)).
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The fnr(false negative rate) of class k.
        """
        return self.fn(k) / (self.fn(k) + self.tp(k))

    def _accuracy_kclass(self, k: int) -> float:
        """
        Get the accuracy of class k, namely (TP(k) + TN(k))/ (TP(k) + TN(k) + FP(k) + FN(k)).
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The accuracy of class k.
        """
        return self.confusion_matrix[k][k] / self.true_num[k]

    def overall_accuracy(self) -> float :
        """
        Get the overall accuracy, namely number of correct predictions / total number
        It is allowed to be called by its alias ``oa`` or ``micro_f1`` .
        Returns:
            The overall accuracy.
        """
        return self.total_true / self.size

    def mean_overall_accuracy(self) -> float:
        """
        Get the mean overall accuracy, namely (Σ(number of class k * accuracy of class k)) over k / total number
        It is allowed to be called by its alias ``moa`` .
        Returns:
            The mean overall accuracy.
        """

        return sum([self.p(k) * self._accuracy_kclass(k) for k in range(self.num_class)]) / self.size


    def accuracy(self, mode: Union[Literal['overall', 'mean'], int] = 'overall'):
        """
        Get the accuracy specified by ``mode``.
        Args:
            mode ('overall' | 'mean' | int, optional): The kind of accurary.

                - ``'overall'`` (default): Get overall accuracy, namely number of correct predictions / total number.

                - ``'mean'``: Get mean overall accuracy, namely (Σ(number of class k * accuracy of class k)) over k / total number

                - int: Get accuracy of class k, namely (TP(k) + TN(k))/ (TP(k) + TN(k) + FP(k) + FN(k)).

        Returns:
            The accuracy specified by ``mode``.
        """
        if mode == 'mean':
            return self.mean_overall_accuracy()
        elif isinstance(mode, int):
            return self._accuracy_kclass(mode)
        return self.overall_accuracy()

    def iou(self, k: int) -> float:
        """
        Get the IoU（intersection-over-union) score of class k, namely TP(k) / (TP(k) + FP(k) + FN(k)).
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The IoU(intersection-over-union) score of class k.
        """
        return self.tp(k) / (self.tp(k) + self.fp(k) + self.fn(k))

    def mean_iou(self) -> float:
        """
        Get the mean IoU（intersection-over-union）score, namely (Σ(IoU(k)) over k) / number of classes.
        It is allowed to be called by its alias ``miou`` .
         Returns:
             The mean IoU（intersection-over-union）score.
        """
        return sum([self.iou(k) for k in range(self.num_class)]) / self.num_class

    def fw_iou(self) -> float:
        """
        Get the frequency-weighted IoU（intersection-over-union）score, namely (Σ(number of class k * IoU(k)) over k) / total number.
        Returns:
            The frequency-weighted IoU（intersection-over-union）score.
        """
        return sum([self.p(k) * self.iou(k) for k in range(self.num_class)]) / self.size

    def _f1_kclass(self, k: int) -> float:
        """
        Get the F1 score of class k, namely 2 * precision(k) * recall(k) / (precision(k) + recall(k)).
        Args:
            k (int): The class number to be analyzed.
        Returns:
            The F1 score of class k.
        """
        precision_and_recall = self.precision(k) + self.recall(k)
        return 2 * self.precision(k) * self.recall(k) / precision_and_recall if precision_and_recall != 0 else 0

    def macro_f1(self) -> float:
        """
        Get the macro F1 score, namely (Σ(F1(k)) over k) / number of classes.
        Returns:
            The macro F1 score.
        """
        return sum([self._f1_kclass(k) for k in range(self.num_class)]) / self.num_class


    def weighted_f1(self):
        """
        Get the weighted F1 score, namely (Σ(number of class k * F1(k)) over k) / total number.
        Returns:
            The weighted F1 score.
        """
        return sum([self.p(k) * self._f1_kclass(k) for k in range(self.num_class)]) / self.size


    def f1(self, mode: Union[Literal['macro', 'micro', 'weighted'], int] = 'macro') -> float:
        """
        Get the F1 score specified by ``mode``.
        Args:
            mode ('macro' | 'micro' | 'weighted' | int, optional): The kind of F1 score.
                - ``'macro'`` (default): Get the macro F1 score, namely (Σ(F1(k)) over k) / number of classes.

                - ``'micro'`` : Get the micro F1 score, namely overall accuracy.

                - ``'weighted'`` : Get the weighted F1 score, namely (Σ(number of class k * F1(k)) over k) / total number.

                - int: Get the F1 score of class k, namely 2 * precision(k) * recall(k) / (precision(k) + recall(k)).
        Returns:
            The F1 score specified by ``mode``.
        """
        if mode == 'micro':
            return self.micro_f1()
        elif mode == 'weighted':
            return self.weighted_f1()
        elif isinstance(mode, int):
            return self._f1_kclass(mode)
        return self.macro_f1()

    def roc(self, k: int, ax: plt.Axes = None) -> tuple[plt.Axes, float]:
        """
         Draw the ROC curve of class k and curve ``y = 1-x``.
         the attribute ``roc_curve`` must be ``True`` when calling this method.
         Raises:
             AssertionError: If the attribute ``roc_curve`` is False.
         Args:
             k (int): The class number the ROC curve of which is to be drawn.
             ax (plt.Axes, optional): The axes to draw the ROC curve.
                If not specified, a new axes will be created.
                The curve ``y = 1-x`` will be also drawn on the axes.
         Notes:
             If one wants to draw the ROC curve of multiple classes at once, he/she should set ``roc_curve=True`` when initializing the instance.
         Returns:
             ( ``ax`` , ``auc``) where ``ax`` is the axes where the ROC curve was drawn and ``auc`` is the area under the ROC curve.
        """
        assert self.roc_curve, 'please use roc_curve=True'
        probs = self.all_probs[:, k].cpu().numpy()
        labels = (self.all_labels == k).cpu().numpy().astype(int)


        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = metrics.auc(fpr, tpr)
        xs = []
        ys = []
        for x, y in zip(fpr, tpr):
            xs.append(x)
            ys.append(y)
        ax = charts.plot_in_one_chart(xs, [ys, [1 - x for x in xs]], 'roc curve', 'y=1-x', title=f'auc={roc_auc:.2f}', marker=None, y_lim='no_limit', show=False, ax=ax)
        return ax, roc_auc

    def auc(self, k: int) -> float:
        """
        Get the AUC(area under the ROC curve) score of class k.
        Args:
            k (int): The class number to be nalyzed.
        Returns:
            The AUC(area under the ROC curve)  score of class k.
        """
        _, auc = self.roc(k)
        return auc

    tpr = sensitivity = recall
    tnr = specificity
    micro_f1 = oa = overall_accuracy
    moa = mean_overall_accuracy
    miou = mean_iou

    def all_metric(self, k: int | None = None, digits: int = 3) -> dict[str, float]:
        """
            Get a dict of all the metrics.
            Args:
                k (int | None, optional): The class number to be analyzed.
                    - None (default): Get all overall metrics,
                      which include metrics as follows:
                      overall acccuracy: it can be got by the key ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
                      mean overall accuracy: it can be got by the key ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
                      mean IoU: it can be got by the key ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
                      frequency-weighted IoU: it can be got by the key ``'FWIoU'`` / ``'fwiou'``;
                      macro F1: it can be got by the key ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
                      weighted F1: it can be got by the key ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;

                    - int: Get all the metrics of class k.
                      which only include metrics as follows:
                      accuracy: it can be got by the key ``'accuracy'``;
                      precision: it can be got by the key ``'precision'``;
                      npv: it can be got by the key ``'npv'``;
                      recall: it can be got by the key ``'recall'`` / ``'sensitivity'`` / ``'TPR'`` / ``'tpr'``;
                      specificity: it can be got by the key ``'specificity'`` / ``'TNR'`` / ``'tnr'``;
                      FNR: it can be got by the key ``'FNR'`` / ``'fnr'``;
                      FPR: it can be got by the key ``'FPR'`` / ``'fpr'``;
                      IoU: it can be got by the key ``'IoU'`` / ``'iou'``;
                      F1: it can be got by the key ``'F1'`` / ``'f1'``;

                digits (int, optional): The number of decimal places to keep. Defaults to 3.
            Returns:
                A dict of all the metrics.
        """
        if k is None:
            oa = round(self.overall_accuracy(), digits)
            moa = round(self.mean_overall_accuracy(), digits)
            miou = round(self.mean_iou(), digits)
            fw_iou = round(self.fw_iou(), digits)
            f1 = round(self.macro_f1(), digits)
            weighted_f1 = round(self.weighted_f1(), digits)
            overall = {
                'accuracy': oa, 'oa': oa, 'overall_accuracy': oa, 'OA': oa,
                'moa': moa, 'MOA': moa, 'mean_accuracy': moa, 'mean_overall_accuracy': moa,
                'miou': miou, 'mIoU': miou, 'mean_iou': miou, 'mean_IoU': miou,
                'FWIoU': fw_iou, 'fwiou': fw_iou,
                'F1': f1, 'f1': f1, 'macro_F1': f1, 'Macro_F1': f1, 'macro_f1': f1,
                'Weighted_F1': weighted_f1, 'Weighted_f1': weighted_f1, 'weighted_f1': weighted_f1
            }
            return overall
        else:
            accurary = round(self.accuracy(k), digits)
            precision = round(self.precision(k), digits)
            npv = round(self.npv(k), digits)
            recall = round(self.recall(k), digits)
            specificity = round(self.specificity(k), digits)
            fnr = round(self.fnr(k), digits)
            fpr = round(self.fpr(k), digits)
            iou = round(self.iou(k), digits)
            f1 = round(self._f1_kclass(k), digits)
            kclass={
                'accuracy': accurary,
                'precision': precision,
                'npv': npv,
                'recall': recall, 'sensitivity': recall, 'TPR': recall, 'tpr': recall,
                'specificity': specificity, 'TNR': specificity, 'tnr': specificity,
                'FNR': fnr, 'fnr': fnr,
                'FPR': fpr, 'fpr': fpr,
                'IoU': iou, 'iou': iou,
                'F1': f1, 'f1': f1,
            }
            return kclass

    def get_metrics(self, metrics: Sequence[str], k: int | None =None, digits: int = 3) -> dict[str,float]:
        """
        Get the metrics specified by ``metrics``.
        Args:
            metrics (Sequence[str]): The names of the metrics to be gotten.
            k (int | None, optional): The class number to be analyzed.
                   - None (default): Get all overall metrics,
                     which include metrics as follows:
                     overall acccuracy: it can be got by the key ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
                     mean overall accuracy: it can be got by the key ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
                     mean IoU: it can be got by the key ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
                     frequency-weighted IoU: it can be got by the key ``'FWIoU'`` / ``'fwiou'``;
                     macro F1: it can be got by the key ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
                     weighted F1: it can be got by the key ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;

                   - int: Get all the metrics of class k.
                     which only include metrics as follows:
                     accuracy: it can be got by the key ``'accuracy'``;
                     precision: it can be got by the key ``'precision'``;
                     npv: it can be got by the key ``'npv'``;
                     recall: it can be got by the key ``'recall'`` / ``'sensitivity'`` / ``'TPR'`` / ``'tpr'``;
                     specificity: it can be got by the key ``'specificity'`` / ``'TNR'`` / ``'tnr'``;
                     FNR: it can be got by the key ``'FNR'`` / ``'fnr'``;
                     FPR: it can be got by the key ``'FPR'`` / ``'fpr'``;
                     IoU: it can be got by the key ``'IoU'`` / ``'iou'``;
                     F1: it can be got by the key ``'F1'`` / ``'f1'``;

            digits (int, optional): The number of decimal places to keep. Defaults to 3
        Returns:
            A dict whose keys are the names of the metrics specified by ``metrics`` and values are the corresponding values.
        """
        metric_dict = self.all_metric(k=k, digits=digits)
        return {self.mode + '_' + m: metric_dict.get(m, 0) for m in metrics}

    def __str__(self):
        """
        The string representation of the instance is the overall metrics, including overall accuracy, mIOU, FWIOU, Macro-F1 and Weighted-F1.
        """
        return (f'accuracy：{self.accuracy():.2f}\t'
                f'mIOU: {self.mean_iou():.2f}\t'
                f'FWIOU: {self.fw_iou():.2f}\t'
                f'Macro-F1: {self.f1():.2f}\t'
                f'Weighted-F1: {self.weighted_f1():.2f}\t')

class ShowEvaluateResult:
    """
    Show the metrics for a classification model, which shows the result analyzed in the class ``ClassificationEvaluate``.
    Args:
        net (nn.Module, optional): The model to be evaluated and shown.
            If not specified, param ``evaluate`` must be specified.
        test_iter (DataLoader[GenericDataset], optional): The dataloader of dataset the model ``net`` based on which must be an obeject of ``GenericDataset``
            If not specified, param ``evaluate`` must be specified.
        num_class (int, optional): The number of classes.
            If not specified, param ``evaluate`` must be specified.
        evaluate (ClassificationEvaluate, optional): A object of ``ClassificationEvaluate`` which needs showing.
            If not specified, param ``net``, ``test_iter`` and ``num_class`` all must be specified.
            If specified, other parameters must keep teir ddefault values and not be allowed to specify .
        class_label (list[str], optional): The label of each class.
            It must not be specified when param ``evaluate`` is specified.
            See ``ClassificationEvaluate.class_label`` for details.
        pixelwise(bool, optional): Whether the labels are pixelwised about features.
            It must not be specified when param ``evaluate`` is specified.
            Default is False.
            See ``ClassificationEvaluate.pixelwise`` for details.
        device (str, optional): The device used for evaluating and showing.
            It must not be specified when param ``evaluate`` is specified.
            Default is 'cuda:0'.
        roc_curve (bool): Whether to calculate the ROC curve.
            It must not be specified when param ``evaluate`` is specified.
            See ``ClassificationEvaluate.roc_curve`` for details.
            Default is False.
        batch_per_roc_sample_point (int): The number of batches used to calculate a single point on the roc curve.
            It must not be specified when param ``evaluate`` is specified.
            See ``ClassificationEvaluate.batch_per_roc_sample_point`` for details.
            Default is 1.
    """
    def __init__(
            self,
            net: nn.Module = None,
            test_iter: DataLoader[Dataset] = None,
            num_class: int = None,
            evaluate: ClassificationEvaluate = None,
            class_label: list[str] = None,
            pixelwise: bool = False,
            device: str = 'cuda:0',
            roc_curve: bool = False,
            batch_per_roc_sample_point: int = 1
    ):
        if evaluate is None:
            assert net is not None and test_iter is not None and num_class is not None , 'net, test_iter, num_class should be specified when evaluate is not specified'
            self.ce = ClassificationEvaluate(
                net,
                test_iter,
                num_class,
                class_label=class_label,
                pixelwise=pixelwise,
                device=device,
                roc_curve=roc_curve,
                batch_per_roc_sample_point=batch_per_roc_sample_point
            )
        else:
            assert (net is None and test_iter is None and num_class is None
                    and class_label is None and pixelwise == False and device == 'cuda:0'
                    and roc_curve == False and batch_per_roc_sample_point == 1) , \
                'other parameters must keep their default values when evaluate is specified'
            self.ce = evaluate

    def __str__(self):
        return self.ce.__str__()

    def print_cm(self, k: int | None = None) -> None:
        """
        Print the confusion matrix.
        Args:
            k (None | int, optional): The class number.

                - None (default): Show the overall confusion matrix.

                - int: Show the confusion matrix of class k.
        """
        cm = self.ce.cm(k)
        for i in range(len(cm)):
            for j in range(len(cm)):
                print(cm[i][j], end=' ')
            print()

    def roc(self, k: int, ax: plt.Axes = None) -> plt.Axes:
        """
        Show the ROC curve of class k and curve ``y = 1-x``.
        If attribute ``evaluate`` is specified, the attribute ``roc_curve`` in this object must be ``True`` when calling this method.
        (but the param``roc_curve`` in the constructor of this object should still keep its default value ``false`` since ``evaluate`` is specfied  ).
        If attribute ``evaluate`` is not specified, param ``roc_curve`` in the constructor of this object must be ``True`` when calling this method.
        Args:
            k (int): The class number the ROC curve of which is to be drawn.
            ax (plt.Axes, optional): The axes to draw the ROC curve.
                If not specified, a new axes will be created.
                The curve ``y = 1-x`` will be also drawn on the axes.
        Returns:
            The axes of the ROC curve.
        """
        ax, auc = self.ce.roc(k, ax=ax)
        ax.set_title(f'ROC of {self.ce.class_label[k]} class\nAUC: {auc:.2f}')
        return ax

    def show_confusion_matrix(self, k: int | None = None, ax :plt.Axes = None) -> plt.Axes:
        """
        Show the confusion matrix as a axes.
        If matrix is large, it may only show partial labels.
        Args:
            k (None | int, optional): The class number.

                - None (default): Show the overall confusion matrix.

                - int: Show the confusion matrix of class k.
            ax (plt.Axes, optional): The axes to draw the confusion matrix.
                If not specified, a new axes will be created.
            Returns:
                The axes of the confusion matrix.
        """
        cm = self.ce.cm(k)
        class_label = self.ce.class_label
        if k is not None:
            class_label = [class_label[k], 'not ' + class_label[k]]
        cm = pd.DataFrame(cm, index=class_label, columns=class_label)
        if ax is None:
            ax = sns.heatmap(cm, annot=False if k is None else True, fmt='d', cmap='coolwarm', annot_kws={"size": 8} )
        else:
            sns.heatmap(cm, annot=False if k is None else True, fmt='d', cmap='coolwarm', ax=ax)
        ax.set_title(f'{self.ce.class_label[k] if k is not None else "overall"} confusion matrix')


        return ax

    def show_all_metrics(
            self,
            k: int | None= None,
            metric: list[str] = None,
            ax:plt.Axes = None
    ) -> plt.Axes:

        """
        Show  all the metrics or specified metrics of the model.

        Args:
            k (None | int, optional): The class number.

                - None (default): Show the overall metrics

                - int: Show the metrics of class k.
                  

            metric (list[str], optional): The metrics to be shown.
                If not specified, all the metrics will be shown, depending on the value of ``k`` .
                The specified metrics, namely the element in the list must be included as follows:
                
                - If ``k`` is ``None``, only metrics in the following can be specified, which only strings enclosed in quotes can be elements of the list
                  overall acccuracy: it can be specified by  ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
                  mean overall accuracy: it can be specified by  ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
                  mean IoU: it can be specified by  ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
                  frequency-weighted IoU: it can be specified by  ``'FWIoU'`` / ``'fwiou'``;
                  macro F1: it can be specified by  ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
                  weighted F1: it can be specified by  ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;
                
                - If ``k`` is a int, only metrics in the following can be specified, which only strings enclosed in quotes can be elements of the list 
                  accuracy: it can be specified by  ``'accuracy'``;
                  precision: it can be specified by  ``'precision'``;
                  npv: it can be specified by  ``'npv'``;
                  recall: it can be specified by  ``'recall'`` / ``'sensitivity'`` / ``'TPR'`` / ``'tpr'``;
                  specificity: it can be specified by  ``'specificity'`` / ``'TNR'`` / ``'tnr'``;
                  FNR: it can be specified by  ``'FNR'`` / ``'fnr'``;
                  FPR: it can be specified by  ``'FPR'`` / ``'fpr'``;
                  IoU: it can be specified by  ``'IoU'`` / ``'iou'``;
                  F1: it can be specified by  ``'F1'`` / ``'f1'``;

            ax (plt.Axes, optional): The axes to draw the metrics.
                If not specified, a new axes will be created.

            Returns:
                The axes of the metrics.
        """
        all_metric = self.ce.all_metric(k=k)
        x = metric
        if x is None:
            if k is None:
                x = ['overall_accuracy', 'mean_accuracy', 'mIoU', 'FWIoU', 'Macro_F1', 'Weighted_F1']
            else:
                x = ['accuracy', 'precision', 'npv', 'recall', 'specificity', 'FNR', 'FPR', 'IoU', 'F1']

        y = [all_metric.get(key) for key in x]
        if ax is None:
            ax = plt.subplot()
        ax.bar(x, y, width=0.8, )
        for key, value in zip(x, y):
            ax.text(key, value, str(value), ha='center', va='bottom')
        ax.set_title(f'all metric of {self.ce.class_label[k] if k is not None else "overall"} ')
        ax.set_ylim(ymax=1.2)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha='center')

        return ax

    def show_metric_figure(
            self,
            klist: list[int | None] = None,
            metric_k: list[str] = None,
            metric_all: list[str] = None,
            save: str = None,
            **kwargs
    ) -> plt.Figure:
        """
        Show the confusion matrix, metrics and roc curve of the model.
        Args:
            klist (list[int | None], optional): The class number to be shown.

                - If a element denoted as k in the list is a int, it means showing the confusion matrix, metrics and roc curve(if ``roc_curve`` is True) of the class k
                  as a row of charts on the returned figure.

                - If a element in the list is ``None``, it means showing overall confusion matrix, overall metrics.
                  as a row of charts on the returned figure.

            metric_k (list[str], optional): The metrics to be shown for one class when one element in the ``klist`` is a class number.
                only metrics in the following can be specified, which only strings enclosed in quotes can be elements of metric_k list.
                If not specified, all of the following metrics will be shown.
                accuracy: it can be specified by  ``'accuracy'``;
                precision: it can be specified by  ``'precision'``;
                npv: it can be specified by  ``'npv'``;
                recall: it can be specified by  ``'recall'`` / ``'sensitivity'`` / ``'TPR'`` / ``'tpr'``;
                specificity: it can be specified by  ``'specificity'`` / ``'TNR'`` / ``'tnr'``;
                FNR: it can be specified by  ``'FNR'`` / ``'fnr'``;
                FPR: it can be specified by  ``'FPR'`` / ``'fpr'``;
                IoU: it can be specified by  ``'IoU'`` / ``'iou'``;
                F1: it can be specified by  ``'F1'`` / ``'f1'``;

            metric_all (list[str], optional): The overall metrics to be shown when one element in the ``klist`` is ``None``.
                only metrics in the following can be specified, which only strings enclosed in quotes can be elements of metric_k list.
                If not specified, all of the following metrics will be shown.
                overall acccuracy: it can be specified by  ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
                mean overall accuracy: it can be specified by  ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
                mean IoU: it can be specified by  ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
                frequency-weighted IoU: it can be specified by  ``'FWIoU'`` / ``'fwiou'``;
                macro F1: it can be specified by  ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
                weighted F1: it can be specified by  ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;

            save (str, optional): The path to save the returned figure.
                If not specified, the figure will not be saved.
            **kwargs: There's no real point in doing that just to be compatible with other methods
        Returns:
            The figure containing the charts, where the number of rows of these charts in the figure is equal to the length of ``klist``.
        """
        if isinstance(klist, int) or klist is None:
            klist = [klist]
        if self.ce.roc_curve and any(item is not None for item in klist):
            fig, axes = plt.subplots(nrows=len(klist), ncols=3, figsize=(14, 8))
        else:
            fig, axes = plt.subplots(nrows=len(klist), ncols=2, figsize=(14, 8))

        if len(klist) == 1:
            axes = np.array([axes])
        for i in range(len(klist)):
            self.show_confusion_matrix(k=klist[i], ax=axes[i, 0])
            label = metric_k if klist[i] is not None else metric_all
            self.show_all_metrics(k=klist[i], metric=label, ax=axes[i, 1])
            if klist[i] is not None and self.ce.roc_curve:
                self.roc(k=klist[i], ax=axes[i, 2])
            if klist[i] is None and axes.shape[1] == 3:
                axes[i, 2].remove()
        fig.tight_layout()
        fig.subplots_adjust()
        if save is not None:
            fig.savefig(save)
        return fig

    def show_result_img(
            self,
            start_idx: int = 0,
            num: int = 10,
            save: str = None,
            **kwargs
    ) -> plt.Figure:
        """
        Show the result of the model.
        If the label is simple, it will show the origin image with the true label and predicted label on its top.
        If the label is pixelwised, it will show the origin feature image, prediction and label in turn;
        Args:
            start_idx (int, optional): The index of the first sample to be shown.
                Default is 0.
            num (int, optional): The number of samples to be shown.
                Default is 10.
            save (str): The path to save the returned figure.
                If not specified, the figure will not be saved.
            **kwargs: There's no real point in doing that just to be compatible with other methods
        Returns:
            The figure of the result of the model.
        """
        evaluate = self.ce
        imgs = []
        titles = []
        k = start_idx + 1
        for origin, output, label in evaluate.get_result(start_idx=start_idx, num=num):
            if evaluate.pixelwise:
                imgs.extend([origin.cpu(), output.cpu(), label.cpu()])
                titles.extend([f'{k}-th feature', f'{k}-th prediction', f'{k}-th label'])
            else:
                imgs.append(origin.cpu())
                titles.append(f'The true label for the {k}-th feature is：{output}\nThe prediction result for the {k}-th feature is：{evaluate.class_label[label]}')
            k += 1

        return img.img_show(imgs, titles, show=False, vmin=0, vmax=evaluate.num_class if evaluate.pixelwise else None, save=save, **kwargs)

    def show_all_figure(self, **kwargs) -> tuple[plt.Figure, plt.Figure]:
        """
        Show the result of the model and the metrics of the model.
        Args:
            **kwargs: Any parameters of method ``show_result_img`` and ``show_metric_figure``.

        Returns:
            A tuple ( ``img``, ``metric`` ), where ``img`` is the figure of the result of the model, and ``metric`` is the figure of the metrics of the model.

        """
        return self.show_result_img(**kwargs), self.show_metric_figure(**kwargs)

class EvaluateModules:
    """
    Evaluate Modules
    Args:
        nets (list[nn.Module] | nn.Module): A model or a list of models to be evaluated.
        device (str): The device to evaluating the model .
    """
    def __init__(self, nets: list[nn.Module] | nn.Module = None, device: str = 'cuda:0'):
        self.nets = nets
        if isinstance(nets, nn.Module):
            self.nets = [nets]
        [net.to(device) for net in self.nets]
        self.device = device

    def visualize_erf(
            self,
            target_layers: list[nn.Module] | nn.Module,
            test_dset: GenericDataset,
            effient_threshold: float = 0.2,
            ax: plt.Axes = None
    ) -> tuple[plt.Axes | list[plt.Axes], float | list[float]]:
        """
        Visualize Effective Receptive Field(ERF) of the ``nets``
        Args:
            target_layers (list[nn.Module] | nn.Module): The layers whose effective receptive field needs to be calculated.
                If it is an object of ``nn.Module``, the attribute ``nets`` of the class instance must contain only one network,
                If it is a list of ``nn.Module`` objects, its length must be the same as the length of the attribute ``nets`` of the class instance,
                which  each layer in the list should correspond to each network.
            test_dset (GenericDataset): The dataset used for testing.
                It must be an instance of ``GenericDataset`` or its subclass.
            effient_threshold (float, optional): The threshold for determining whether a pixel is considered efficient.
                Default is 0.2.
            ax (plt.Axes, optional): The axes to plot the heatmap.
                If not provided, a new axes will be created.

        Returns:
            - A tuple(``ax`` , ``energy_density`` ) if the attribute ``nets`` of the class instance contains only one network
              where ``ax`` is an instance of ``plt.Axes``  containing the heatmap of the effective receptive field of the layer specified by ``target_layer``
              and ``energy_density`` is an a float roundes to three decimal places representing the percentage of efficient pixels in the effective receptive field.
            - A tuple(axes, energy_densities) if the attribute ``nets`` of the class instance contains more than one network,
              where ``axes`` is a list of instances of ``plt.Axes`` containing the heatmaps of the effective receptive field of each network,
              and ``energy_densities`` is a list of floats rounded to three decimal places
              representing the percentage of efficient pixels in the effective receptive field of each network.

        """
        axes = []
        energy_densities = []
        device = self.device
        if isinstance(target_layers, nn.Module):
            target_layers = [target_layers]
        assert len(target_layers) == len(self.nets), 'The length of target_layers must be equal to the length of nets.'
        for i in range(len(self.nets)):
            net=self.nets[i]
            net=net.to(device)
            net.eval()

            activations = []

            def forward_hook(module, inp, out):
                activations.append(out)

            handle = target_layers[i].register_forward_hook(forward_hook)
            input_size = test_dset.target_size
            used_images = 0
            imgs = DataLoader(test_dset, batch_size=10, shuffle=False, drop_last=True)
            erf = torch.zeros((1, 10, input_size[0], input_size[1]), device=device)
            for img in imgs:
                img = img.clone().detach().requires_grad_(True)
                used_images+=1
                net.zero_grad()
                activations.clear()
                _ = net(img)
                feat = activations[0]

                _, c, h, w = feat.shape
                center_h, center_w = h // 2, w // 2
                target = feat[:, :, center_h, center_w].sum()
                target.backward()
                grad = img.grad.detach().abs()

                grad = grad.sum(dim=1)
                grad = torch.nn.functional.interpolate(
                    grad.unsqueeze(0),
                    size=input_size,
                    mode="bilinea",
                    align_corners=False
                )
                erf += grad
            handle.remove()
            erf = erf.sum(dim=1).squeeze(0)
            erf /= used_images

            erf = erf.cpu().numpy()
            erf = erf / (erf.max()) if erf.max() != 0 else erf

            if ax is None:
                ax = sns.heatmap(erf, fmt='.2f', cmap='coolwarm')
            else:
                sns.heatmap(erf, cmap='coolwarm', ax=ax)
            ax.set_title("Effective Receptive Field")
            ax.axis("off")
            mask = erf > effient_threshold
            axes.append(ax)
            energy_densities.append(int(np.sum(erf[mask])) / np.sum(mask))
        return (axes if len(axes) > 1 else axes[0],
                [round(e, 3) for e in energy_densities] if len(energy_densities) > 1 else round(energy_densities[0], 3))

    def monitor_metric_by_datasets(
            self,
            datasets: GenericDataset | list[GenericDataset],
            monitors: str | list[str],
            batch_size: int = 4,
            content_in_one_chart: Literal["net", "nets", "n", "datasets","dataset", "dset", "dsets", "d"] = 'net',
            k: str | None = None,
            x_label: str | list[str] = None,
            x_axis: str = None
    ) -> plt.Figure:
        """

        Evaluate and plot specified metrics for networks across multiple datasets.

        This method evaluates one or more metrics on specified datasets for all networks stored in ``self.nets``,
        and creates line charts to visualize the results. The organization of the charts can be controlled
        by the ``content_in_one_chart`` parameter.

        Args:
            datasets (GenericDataset | list[GenericDataset]): A single dataset or a list of datasets to be evaluated.
                Each dataset must be an instance of GenericDataset.

            monitors (str | list[str]): A single metric name or a list of metric names to be monitored.

                - If param ``k`` is None, only metrics in the following can be specified:
                  overall acccuracy: it can be specified by  ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
                  mean overall accuracy: it can be specified by  ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
                  mean IoU: it can be specified by  ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
                  frequency-weighted IoU: it can be specified by  ``'FWIoU'`` / ``'fwiou'``;
                  macro F1: it can be specified by  ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
                  weighted F1: it can be specified by  ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;

                - If param ``k`` is a class number, only metrics in the following can be specified:
                  accuracy: it can be specified by  ``'accuracy'``;
                  precision: it can be specified by  ``'precision'``;
                  npv: it can be specified by  ``'npv'``;
                  recall: it can be specified by  ``'recall'`` / ``'sensitivity'`` / ``'TPR'`` / ``'tpr'``;
                  specificity: it can be specified by  ``'specificity'`` / ``'TNR'`` / ``'tnr'``;
                  FNR: it can be specified by  ``'FNR'`` / ``'fnr'``;
                  FPR: it can be specified by  ``'FPR'`` / ``'fpr'``;
                  IoU: it can be specified by  ``'IoU'`` / ``'iou'``;
                  F1: it can be specified by  ``'F1'`` / ``'f1'``;


            content_in_one_chart ("net", "nets", "n", "datasets", "dataset", "dset", "dsets", "d", optional):
                Controls how the data is organized in the charts. Default is 'net'.

                - 'net' mode, namely ``'net'``, ``'nets'``, or ``'n'``: Creates one chart per network, showing all datasets
                  on the x-axis and the specified metrics as lines. Useful for comparing a single network's
                  performance across different datasets
                  .
                - 'dataset' mode, namely ``'datasets'``, ``'dset'``, ``'dsets'``, or ``'d'``: Creates one chart per dataset,
                  showing all networks on the x-axis and the specified metrics as lines. Useful for
                  comparing different networks' performance on a specific dataset.

            batch_size (int, optional): The batch size of datasets used when evaluating.
                Default is 4.

            k (str | None, optional): The class number.

                - None (default): Show the overall metrics, in which case the param ``monitors`` has to specified overall metrics.

                - int: Show the metrics of class k, in which case the param ``monitors`` has to specified the metrics of class k.

            x_label (list[str] | str, optional): The labels for the x-axis.
                If not provided, the label will be the name of dataset for 'dataset' mode or the name of network class for 'net' mode.

                - list[str]: It will be x_label in each chart.
                  it must have the same length as the number of datasets for 'dataset' mode or the number of networks for 'net' mode.

                - str: It will be common x_label in all charts.

            x_axis (str, optional): The data of x-axis data.
                If not provided, the x-axis will be the index of dataset for 'net' mode or  the index of network index for 'dataset' mode.
                If provided, it must have the same length as the number of datasets for 'net' mode or the number of networks for 'datasets' mode.

        Returns:
            The figure containing the charts whose number is the same as the number of dataset for 'dataset' mode or the number of network for 'net' mode.
        """
        device = self.device
        if isinstance(datasets, GenericDataset):
            datasets = [datasets]
        if isinstance(monitors, str):
            monitors = [monitors]
        num_metric = len(monitors)
        if x_axis is None:
            x = list(range(len(datasets))) if content_in_one_chart in ['net', 'nets', 'n'] else list(range(len(self.nets)))
        else:
            x = x_axis
        fig = None
        if content_in_one_chart in ['datasets', 'dataset', 'dset', 'dsets', 'd']:
            fig, axes = plt.subplots(len(datasets), nrows=len(datasets), ncols=1)
            if isinstance(x_label, str):
                x_label = [x_label] * len(datasets)
            for i in range(len(datasets)):
                dset = datasets[i]
                y = [[] for _ in range(num_metric)]
                for net in self.nets:
                    num_class = dset.num_class
                    test_dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
                    evaluate = ClassificationEvaluate(test_iter=test_dataloader, net=net, device=device, num_class=num_class)
                    metrics_dict:dict[str, float] = evaluate.all_metric(k=k)
                    for j in range(num_metric):
                        monitor_metric = metrics_dict.get(monitors[j], None)
                        assert monitor_metric is not None, f' metric {monitors[j]} is not supported'
                        y[j].append(monitor_metric)
                axes = charts.plot_in_one_chart(x, y, *monitors, ax=axes[i])
                axes.set_xlabel(x_label[i] if x_label[i] is not None else dset.name)
                axes.legend()
        elif content_in_one_chart in ['net', 'nets', 'n']:
            fig, axes = plt.subplots(len(self.nets), nrows=len(self.nets), ncols=1)
            if isinstance(x_label, str):
                x_label = [x_label] * len(self.nets)
            for i in range(len(self.nets)):
                net = self.nets[i]
                y = [[] for _ in range(num_metric)]
                for dset in datasets:
                    num_class = dset.num_class
                    test_dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
                    evaluate = ClassificationEvaluate(test_iter=test_dataloader, net=net, device=device, num_class=num_class)
                    metrics_dict:dict[str, float] = evaluate.all_metric(k=k)
                    for j in range(num_metric):
                        monitor_metric = metrics_dict.get(monitors[j], None)
                        assert monitor_metric is not None, f' metric {monitors[j]} is not supported'
                        y[j].append(monitor_metric)
                axes = charts.plot_in_one_chart(x, y, *monitors, ax=axes[i])
                axes.set_xlabel(x_label[i] if x_label[i] is not None else net.__class__.__name__)
                axes.legend()

        return fig

    def flops_and_parms(self, size=(3, 256, 256), print_per_layer=False) -> tuple[str, str] | list[tuple[str, str]]:
        """
        Calculate the FLOPs and Parameters of the ``nets``.
        Args:
            size (tuple, optional): The input size of the model.
                Default is (3, 256, 256).
            print_per_layer (bool, optional): Whether to print the FLOPs and Parameters per layer.
                Default is False.
        Returns:
            - A tuple ( ``flops`` , ``params`` ) if the attribute ``nets`` of the class instance contains only one network
              where ``flops`` is the total number of floating-point operations performed by the network on the given input size
              and ``params`` is the total number of trainable parameters in the network.
            - A list of tuples ( ``flops`` , ``params`` ) if the attribute ``nets`` of the class instance contains more than one network
              where ``flops`` are the total numbers of floating-point operations performed by each network on the given input size
              and ``params`` are the total numbers of trainable parameters in each network respectively.
        """
        res = []
        x = torch.randn(size).to(self.device)
        for net in self.nets:
            flops, params = get_model_complexity_info(model=net, input_res=size, print_per_layer_stat=print_per_layer)
            res.append((flops, params))
        return res[0] if len(res) == 1 else res