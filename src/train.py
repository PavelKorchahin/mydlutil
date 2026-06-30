# train module
from typing import Literal, Sequence, Any
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from . import plot_in_one_chart, ClassificationEvaluate
from .accumulator import Accumulator
from .model_config import save_model
from pathlib import Path
from torch.amp import autocast, GradScaler
from .config import is_accelerate



def train_epoch(
        net: nn.Module,
        train_iter: DataLoader,
        loss: nn.Module,
        trainer: Optimizer,
        epoch: int,
        device: str = 'cuda:0',
        pixelwise: bool = False,
        lr_scheduler: LRScheduler | Any = None,
        lr_scheduler_type: Literal['epoch', 'batch']='epoch'
) -> float:
    """
    train one epoch
    Args:
        net (torch.nn.Module): Model to be trained.
        train_iter (torch.utils.data.DataLoader): Training data loader which is an object of ``torch.utils.data.DataLoader`` .
        loss (torch.nn.Module): Loss function that take model's output and label as input and return the loss scalar.
        trainer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number,starting from 0.
        device (str, optional): Device to train on.
            Default is ``'cuda:0'`` .
        pixelwise (bool, optional): Whether the label is pixelwised.
            Default is ``False`` .
        lr_scheduler (LRScheduler | Any, optional): Learning rate scheduler.
            If not specified, there is no learning rate scheduler.
        lr_scheduler_type (Literal['epoch', 'batch'], optional): When to update learning rate scheduler.

            - ``'epoch'`` (default) : update learning rate scheduler after epoch.

            - ``'batch'`` : update  learning rate scheduler after each batch iteration.

    Returns:
        Average training loss value of this epoch.
    """
    net = net.to(device)
    net.train()
    metric = Accumulator(2)
    if is_accelerate():
        torch.set_float32_matmul_precision('high')
    it = tqdm(train_iter, leave=False, desc=f'training epoch {epoch + 1}: ')
    scale = GradScaler()
    for x, y in it:
        if x is None or y is None:
            continue
        trainer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        if pixelwise:
            y = y.squeeze(dim=1).to(device)
        if is_accelerate():
            with autocast('cuda:0'):
                y_hat = net(x)
                l = loss(y_hat, y)
            scale.scale(l).backward()
            scale.step(trainer)
            scale.update()
        else:
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
        with torch.no_grad():
            metric.add(float(l) * y.shape[1] * y.shape[2] if pixelwise else float(l) * len(y) , y.numel())
        if lr_scheduler is not None and lr_scheduler_type == 'batch':
            lr_scheduler.step()
    it.close()
    if lr_scheduler is not None and lr_scheduler_type == 'epoch':
        lr_scheduler.step()
    return metric[0] / metric[1]
def train(
        net: nn.Module,
        num_class: int,
        train_iter: DataLoader,
        test_iter: DataLoader,
        loss: nn.Module,
        num_epochs: int,
        trainers: Optimizer | Sequence[Optimizer],
        trainers_epoch_threshold: Sequence[int] | int = (0, ),
        device: str = 'cuda:0',
        lr_scheduler: LRScheduler | Any = None,
        lr_scheduler_type: Literal['epoch', 'batch']='epoch',
        pixelwise: bool = False,
        loss_curve_save: str | Path = None,
        metrics: Sequence[str] = ('accuracy', 'miou'),
        model_save: str | Path = None,
        log_save: str | Path = None,
        log_prompt: str = 'logs',
        save_query: bool = False,
        early_stopping_metric: str = None,
        early_stopping_times: int = 7,
        early_stopping_mode: Literal["max", "min"]='max',
        early_stopping_min_delta: float = 0.00
) -> tuple[plt.Axes, int, int]:
    """
    train model
    Args:
        net (torch.nn.Module): Model to be trained.
        num_class (int): The number of classes.
        train_iter (torch.utils.data.DataLoader): Training data loader which is an object of ``torch.utils.data.DataLoader`` .
        test_iter (torch.utils.data.DataLoader): Testing data loader which is an object of ``torch.utils.data.DataLoader`` .
        loss (torch.nn.Module): Loss function that take model's output and label as input and return the loss scalar.
        num_epochs (int): The number of epochs to train.
        trainers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer]): Optimizer or sequence of optimizers.
            Used to use different optimizers in different stages of training.
            It can be a single optimizer or a sequence of optimizers.
        trainers_epoch_threshold (Sequence[int] | int, optional): The epoch threshold when to change optimizer.
            Default is ``(0, )`` .

            - Sequence[int]: Specify the epoch numbers to start using a new optimizer.
              For example, if trainers_epoch_threshold is ``(0, 10, 15)``,
              it means that the number of optimizers is 3 given by ``trainers`` ,
              and the first optimizer is used from epoch 0 to 9,
              the second optimizer is used from epoch 10 to 14,
              and the third optimizer is used from epoch 15 to the end of training.
              The sequence must be a ascending order and start with 0,
              and the length of the squence must be the same as the length of trainers.

            - int: If specified by a int ``n``, it is equivalent to ``(0, n)``
              in which case there are two optimizers given by ``trainers`` .

        device (str, optional): Device to train on.
            Default is ``'cuda:0'`` .
        lr_scheduler (LRScheduler | Any, optional): Learning rate scheduler.
            If not specified, there is no learning rate scheduler.
        lr_scheduler_type (Literal['epoch', 'batch'], optional): When to update learning rate scheduler.

            - ``'epoch'`` (default) : update learning rate scheduler after epoch.

            - ``'batch'`` : update  learning rate scheduler after each batch iteration.

        pixelwise (bool, optional): Whether the label is pixelwised.
            Default is ``False`` .
        loss_curve_save (str | Path, optional): an absolute Path of a img file to save loss curve.
            If not specified, the loss curve will not be drawn and saved.
        metrics (Sequence[str], optional):Names of overall metrics to evaluate on the test dataset.
            Default is ``('accuracy', 'miou')`` .
            The available metrics are as follows:
            overall acccuracy: it can be got by the key ``'accuracy'`` / ``'oa'`` /  ``'OA'`` / ``'overall_accuracy'``;
            mean overall accuracy: it can be got by the key ``'moa'`` / ``'MOA'`` /``'mean_overall_accuracy'`` / ``'mean_accuracy'``;
            mean IoU: it can be got by the key ``'miou'`` / ``'MIoU'`` / ``'mean_iou'`` / ``'mean_IoU'``;
            frequency-weighted IoU: it can be got by the key ``'FWIoU'`` / ``'fwiou'``;
            macro F1: it can be got by the key ``'f1'`` / ``'F1'`` / ``'macro_f1'`` / ``'Macro_F1'``;
            weighted F1: it can be got by the key ``'Weighted_F1'`` / ``'Weighted_f1'`` / ``'weighted_f1'``;
        model_save (str | Path, optional): An absolute path or a relative path to ``config.RESULT_MODEL_DIR`` of a .pth file to save model.
            If not specified, the model will not be saved.
        log_save (str | Path, optional): an absolue Path to save training logs.
            If not specified, the logs will not be saved.
        log_prompt (str, optional): Prompt string to show before logs of each training process.
            It is used to record the hyper-parameters of each training process in the logs file specified by ``log_save`` .
            Default is ``'logs'`` .
        save_query (bool, optional): Whether to query whether to save model, log and loss curve if they are specified respeectively.

            - ``False`` (default): Save model, log and loss curve directly if their paths are specified respectively, without querying and prompting.

            - ``True``: Query user whether to save model, log and loss curve if they are specified respectively.
              Only if one type ``'y'``, the model, log and loss curve will be saved.
        early_stopping_metric (str, optional): Metric name or ``'loss'`` to monitor during training when one needs to implement early stopping strategy.
            If not specifed, the training will not stop until the number of epochs reaches the value of ``num_epochs``.
            Otherwise, the training will stop when the metric will not get any better within ``early_stopping_times`` consecutive epochs.
            This parameter must be in ``metrics`` or ``'loss'`` .
        early_stopping_times (int, optional): Number of times the ``early_stopping_metric`` has to get better before stopping the training.
            Default is ``7`` .
        early_stopping_mode ('max', 'min', optional): Mode of early stopping for ``early_stopping_metric``.

            - ``'max'`` (default): The larger the ``early_stopping_metric``, the better,such as ``'miou'`` / ``'accuracy'``.

            - ``'min'`` : The smaller the ``early_stopping_metric``, the better,such as ``'loss'`` .

        early_stopping_min_delta (float, optional):
            Minimum delta between ``early_stopping_metric`` of the current epoch and the previous best metric
            to consider whether the ``early_stopping_metric`` is getting better.
            Default is ``0.00`` .
    Returns:
        (``ax`` , ``epoch_num``, ``best_epoch_num``)
        where ``ax`` is the plot of loss curve and overall metric given by ``metrics`` curve if the plot is saved or is None if not,
        ``epoch_num`` is the actual number of epochs actually trained,
        and ``best_epoch_num`` is the epoch number at which the best performance was obtained.


    """
    test_metrics = {}
    best_metric = 0 if early_stopping_mode == 'max' else 100000
    best_net_dict = None
    best_epoch_num = 0
    stop_times = 0
    epoch_num = num_epochs
    net = net.to(device)
    logs=[]
    exit_flag = False
    if isinstance(metrics, str):
        metrics = (metrics, )
    if not isinstance(trainers, Sequence):
        trainers = (trainers, )
    if isinstance(trainers_epoch_threshold, int):
        trainers_epoch_threshold = (0, trainers_epoch_threshold)
    assert len(trainers)  == len(trainers_epoch_threshold), (f'the length of trainers_epoch_threshold should be the same as the length of trainers, '
                                                             f'but the length of trainers is {len(trainers)} while '
                                                             f'the length of trainers_epoch_threshold={len(trainers_epoch_threshold)} ')
    assert trainers_epoch_threshold[0] == 0 , 'the beginning of trainers_epoch_threshold should be 0 '
    assert early_stopping_metric is None or early_stopping_metric in metrics or early_stopping_metric == 'loss', 'early_stopping_metric must be in metrics or loss'
    x, y = None, None
    trainers_idx = -1
    if loss_curve_save is not None:
        x = list(range(num_epochs))
        y = [[] for _ in range(len(metrics) + 1 )]
    try:
        for epoch in range(num_epochs):
            if epoch in trainers_epoch_threshold:
                trainers_idx += 1
            avg_loss = train_epoch(
                net=net,
                train_iter=train_iter,
                loss=loss,
                trainer=trainers[trainers_idx],
                device=device,
                epoch=epoch,
                pixelwise=pixelwise,
                lr_scheduler=lr_scheduler,
                lr_scheduler_type=lr_scheduler_type
            )
            test_metrics = ClassificationEvaluate(
                num_class=num_class,
                net=net,
                test_iter=test_iter,
                device=device,
                mode='test'
            ).get_metrics(metrics=metrics)
            torch.cuda.empty_cache()
            k = 1
            if loss_curve_save is not None:
                y[0].append(avg_loss)
                for test_metric in test_metrics.values():
                    y[k].append(test_metric)
                    k += 1
            log = f'epoch{epoch + 1}: avg_loss:{avg_loss:.3f}; {"; ".join([f"{key}:{value:.3f}" for key, value in test_metrics.items()])}'
            print('\b'*100, log, sep='')
            if log_save is not None:
                logs.append(log)
            if early_stopping_metric is not None:
                if early_stopping_metric == 'loss':
                    early_stopping_metric_value = avg_loss
                else:
                    early_stopping_metric_value = test_metrics.get('test_' + early_stopping_metric, None)
                assert early_stopping_metric_value is not None, 'early stopping metric is not exist'
                if early_stopping_mode == 'max':
                    if best_metric + early_stopping_min_delta < early_stopping_metric_value:
                        best_epoch_num = epoch + 1
                        best_metric = early_stopping_metric_value
                        best_net_dict = net.state_dict()
                        stop_times = 0
                    else:
                        stop_times += 1
                        print(f'metric is not improved: {stop_times} times({stop_times} / {early_stopping_times}) the best metric is {best_metric:.3f}')
                elif early_stopping_mode == 'min':
                    if best_metric - early_stopping_min_delta > early_stopping_metric_value:
                        best_epoch_num = epoch + 1
                        best_metric = early_stopping_metric_value
                        best_net_dict = net.state_dict()
                        stop_times = 0
                    else:
                        stop_times += 1
                        print(f'metric is not improved: {stop_times} times({stop_times} / {early_stopping_times})， the best metric is {best_metric:.3f}')
                if stop_times >= early_stopping_times:
                    epoch_num = epoch + 1
                    log = (f'early stopping! {stop_times} times, train epoch num:{epoch_num}, '
                     f'the best epoch num:{best_epoch_num}, the best {early_stopping_metric} is {best_metric:.3f}')
                    print(log)
                    if log_save is not None:
                        logs.append(log)
                    break
    except KeyboardInterrupt:
        print('interrupted!, saving...')
        exit_flag = True
    finally:
        if early_stopping_metric is not None:
            log =  f'end! train epoch num:{epoch_num}, the best epoch num:{best_epoch_num}, the best {early_stopping_metric} is {best_metric:.3f}'
            print(log)
            if log_save is not None:
                logs.append(log)

        return_flag = False
        if save_query:
            try:
                prmopt = input("Do you want to ave? [y/n]").lower()
                if prmopt != 'y':
                    return_flag = True
            except KeyboardInterrupt:
                return_flag = True

        if return_flag:
            return None, epoch_num, best_epoch_num
        if log_save is not None:
            log_save = Path(log_save)
            log_save.parent.mkdir(parents=True, exist_ok=True)
            with open(log_save, 'a+') as f:
                f.writelines('\n')
                f.writelines(log_prompt + "\n")
                for log in logs:
                    f.writelines(log + '\n')
        if model_save is not None:
            save_model(best_net_dict, model_save)
        ax = None
        if loss_curve_save is not None:
            ax = plot_in_one_chart(x, y, *(['avg_loss']  + [key for key in test_metrics.keys()]), show=False, save=loss_curve_save)
    if exit_flag:
        exit(0)
    return ax, epoch_num, best_epoch_num





