# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import logging as logger
import torch

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from vlmo.modules.objectives import compute_irtr_recall
from vlmo.gadgets.my_metrics import Accuracy, VQAScore, Scalar
from vlmo.modules.vlmo_file_check import file_check, ErrorCode


WEIGHT_DECAY = "weight_decay"
PARAMS_CONST = "params"
LR_CONST = "lr"



def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itc":
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

                setattr(pl_module, f"{split}_{k}_vl_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_logit_scale", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        ittr1 = compute_irtr_recall(pl_module, split="val")
        val_ir_r1 = ittr1.val_ir_r1
        val_ir_r5 = ittr1.val_ir_r5
        val_ir_r10 = ittr1.val_ir_r10
        val_tr_r1 = ittr1.val_tr_r1
        val_tr_r5 = ittr1.val_tr_r5
        val_tr_r10 = ittr1.val_tr_r10
        val_avg = (val_ir_r1.item() + val_ir_r5.item() + val_ir_r10.item() +
                   val_tr_r1.item() + val_tr_r5.item() + val_tr_r10.item()) / 6.0
        pl_module.logger.experiment.add_scalar(
            "recalls/val_avg", val_avg, pl_module.global_step
        )
        irtr = compute_irtr_recall(pl_module, split="test")
        ir_r1 = irtr.ir_r1
        ir_r5 = irtr.ir_r5
        ir_r10 = irtr.ir_r10
        tr_r1 = irtr.tr_r1
        tr_r5 = irtr.tr_r5
        tr_r10 = irtr.tr_r10
        test_avg = (ir_r1.item() + ir_r5.item() + ir_r10.item() + tr_r1.item() + tr_r5.item() + tr_r10.item()) / 6.0
        pl_module.logger.experiment.add_scalar(
            "recalls/test_avg", test_avg, pl_module.global_step
        )

        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += val_avg


    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value_dev = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value_dev)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value_test = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value_test)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
                value = value_dev
        elif loss_name == "irtr":
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()

            value = value_i2t + value_t2i
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itc":
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            value_vl_i2t = getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_i2t_accuracy_epoch", value_vl_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").reset()
            
            value_vl_t2i = getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_t2i_accuracy_epoch", value_vl_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").reset()

            value = value_i2t + value_t2i
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k 
        for k, v in pl_module.hparams.config["loss_names"].items() 
        if v >= 1
    ]
    return


def set_select_parm(pl_module, no_decay, head_names, tmp_contains_no_decay, tmp_contains_head_name):
    selected_params = []
    selected_params1 = []
    selected_params2 = []
    selected_params3 = []
    selected_params4 = []
    for n, p in pl_module.named_parameters():
        contains_no_decay = False
        for nd in no_decay:
            if nd in n:
                contains_no_decay = True
                break
        
        contains_head_name = False
        for bb in head_names:
            if bb in n:
                contains_head_name = True
                break
        if contains_no_decay and not contains_head_name:
            selected_params2.append(p)
        elif not contains_no_decay and not contains_head_name:
            selected_params1.append(p)
        elif not contains_no_decay and contains_head_name:
            selected_params3.append(p)
        elif contains_no_decay and contains_head_name:
            selected_params4.append(p)
    if not tmp_contains_no_decay and not tmp_contains_head_name:
        return selected_params1
    elif tmp_contains_no_decay and not tmp_contains_head_name:
        return selected_params2
    elif not tmp_contains_no_decay and tmp_contains_head_name:
        return selected_params3
    elif tmp_contains_no_decay and tmp_contains_head_name:
        return selected_params4
    return selected_params



def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config[WEIGHT_DECAY]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            PARAMS_CONST: set_select_parm(pl_module, no_decay, head_names, False, False),
            WEIGHT_DECAY: wd,
            LR_CONST: lr,
        },
        {
            PARAMS_CONST: set_select_parm(pl_module, no_decay, head_names, True, False),
            WEIGHT_DECAY: 0.0,
            LR_CONST: lr,
        },
        {
            PARAMS_CONST: set_select_parm(pl_module, no_decay, head_names, False, True),
            WEIGHT_DECAY: wd,
            LR_CONST: lr * lr_mult,
        },
        {
            PARAMS_CONST: set_select_parm(pl_module, no_decay, head_names, True, True),
            WEIGHT_DECAY: 0.0,
            LR_CONST: lr * lr_mult,
        },
    ]

    if optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        logger.error("optim_type must be set with adam or sgd",
                    extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise ValueError("optim_type must be set with adam or sgd")

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
