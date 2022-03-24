"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging

from methods.bic import BiasCorrection
from methods.finetune import Finetune
from methods.gdumb import GDumb
from methods.rainbow_memory import RM
from methods.icarl import ICaRL
from methods.joint import Joint
from methods.regularization import EWC, RWalk

logger = logging.getLogger()


def select_method(args, criterion, device, labeled_transform, unlabeled_transform, test_transform, n_classes, transform_train, transform_test):
    kwargs = vars(args)
    if args.mode == "finetune":
        method = Finetune(
            criterion=criterion,
            device=device,
            transform_labeled=labeled_transform,
            transform_unlabeled=unlabeled_transform,
            transform_test=test_transform,
            n_classes=n_classes,
            train_transform=transform_train,
            test_transform=transform_test,
            **kwargs,
        )
    elif args.mode == "rm":
        method = RM(
            criterion=criterion,
            device=device,
            transform_labeled=labeled_transform,
            transform_unlabeled=unlabeled_transform,
            transform_test=test_transform,
            n_classes=n_classes,
            train_transform=transform_train,
            test_transform=transform_test,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [finetune, gdumb]")

    logger.info("CIL Scenario: ")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")

    return method
