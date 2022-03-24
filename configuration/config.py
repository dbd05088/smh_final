"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import argparse


def base_parser():
    parser = argparse.ArgumentParser(
        description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        help="CIL methods [joint, rwalk, icarl, rm,  gdumb, ewc, bic]",
    )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default=None,
        help="memory management [default, random, reservoir, uncertainty, prototype]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default="5",
                        help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=2, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=1,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    parser.add_argument(
        "--stream_env",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="the restriction whether to keep streamed data or not",
    )

    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved. Only for local-machine",
    )

    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet32", help="[resnet18, resnet32]"
    )
    parser.add_argument("--pretrain", action="store_true",
                        help="pretrain model or not")

    # SPR argument
    parser.add_argument(
        '--random_seed', type=int, default=0)
    parser.add_argument(
        '--configs', default='configs/ccmodel-coco.yaml'
    )
    parser.add_argument(
        '--eps', default='episodes/coco-split.yaml'
    )
    parser.add_argument('--log-dir', '-l')
    parser.add_argument('--resume-ckpt', default=None)
    parser.add_argument('--override', default='')

    # Train
    parser.add_argument("--opt_name", type=str,
                        default="sgd", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str,
                        default="cos", help="[cos, anneal]")
    parser.add_argument("--batchsize", type=int,
                        default=128, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=30, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=0,
                        help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--initial_annealing_period",
        type=int,
        default=20,
        help="Initial Period that does not anneal",
    )
    parser.add_argument(
        "--annealing_period",
        type=int,
        default=20,
        help="Period (Epochs) of annealing lr",
    )
    parser.add_argument(
        "--learning_anneal", type=float, default=10, help="Divisor for annealing"
    )
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )
    parser.add_argument(
        "--joint_acc",
        type=float,
        default=0.0,
        help="Accuracy when training all the tasks at once",
    )
    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=[],
        help="Additional train transforms [cotmix, cutout, randaug]",
    )

    # Benchmark
    parser.add_argument("--exp_name", type=str, default="",
                        help="[disjoint, blurry]")

    # ICARL
    parser.add_argument(
        "--feature_size",
        type=int,
        default=2048,
        help="Feature size when embedding a sample",
    )

    # BiC
    parser.add_argument(
        "--distilling",
        action="store_true",
        help="use distilling loss with classification",
    )

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="vr",
        choices=["vr", "vr1", "vr_randaug", "loss"],
        help="A type of uncertainty metric",
    )

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Turn on Debug mode")

    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--data-path', default='./data',
                        type=str, help='data path')
    parser.add_argument('--save-path', default='./checkpoint',
                        type=str, help='save path')
    '''
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'], help='dataset name')
    '''
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=10000,
                        type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1000, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-step', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    '''
    parser.add_argument('--workers', default=4, type=int,
                        help='number of workers')
    '''
    parser.add_argument('--num-classes', default=10,
                        type=int, help='number of classes')
    parser.add_argument('--resize', default=32, type=int, help='resize image')
    parser.add_argument('--batch-size', default=64,
                        type=int, help='train batch size')
    parser.add_argument('--teacher-dropout', default=0,
                        type=float, help='dropout on last dense layer')
    parser.add_argument('--student-dropout', default=0,
                        type=float, help='dropout on last dense layer')
    parser.add_argument('--teacher_lr', default=0.01,
                        type=float, help='train learning late')
    parser.add_argument('--student_lr', default=0.01,
                        type=float, help='train learning late')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='SGD Momentum')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov')
    parser.add_argument('--weight-decay', default=0,
                        type=float, help='train weight decay')
    parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
    parser.add_argument('--warmup-steps', default=0,
                        type=int, help='warmup steps')
    parser.add_argument('--student-wait-steps', default=0,
                        type=int, help='warmup steps')
    parser.add_argument('--grad-clip', default=1e9, type=float,
                        help='gradient norm clipping')
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate model on validation set')
    parser.add_argument('--finetune', action='store_true',
                        help='only finetune model on labeled dataset')
    parser.add_argument('--finetune-epochs', default=625,
                        type=int, help='finetune epochs')
    parser.add_argument('--finetune-batch-size', default=512,
                        type=int, help='finetune batch size')
    parser.add_argument('--finetune-lr', default=3e-5,
                        type=float, help='finetune learning late')
    parser.add_argument('--finetune-weight-decay', default=0,
                        type=float, help='finetune weight decay')
    parser.add_argument('--finetune-momentum', default=0.9,
                        type=float, help='finetune SGD Momentum')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--label-smoothing', default=0,
                        type=float, help='label smoothing alpha')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--threshold', default=0.95,
                        type=float, help='pseudo label threshold')
    parser.add_argument('--temperature', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--uda-steps', default=1, type=float,
                        help='warmup steps of lambda-u')
    parser.add_argument("--randaug", nargs="+", type=int,
                        help="use it like this. --randaug 2 10")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision")
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()
    return args
