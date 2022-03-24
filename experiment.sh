#/bin/bash

# CIL CONFIG
MODE="rm" # joint, gdumb, icarl, rm, ewc, rwalk, bic
# "default": If you want to use the default memory management method.
MEM_MANAGE="uncertainty" # default, random, reservoir, uncertainty, prototype.
RND_SEED=1
DATASET="cifar10" # mnist, cifar10, cifar100, imagenet
STREAM="online" # offline, online
EXP="blurry10" # disjoint, blurry10, blurry30
MEM_SIZE=500 # cifar10: k={200, 500, 1000}, mnist: k=500, cifar100: k=2,000, imagenet: k=20,000
TRANS="cutmix autoaug" # multiple choices: cutmix, cutout, randaug, autoaug

N_WORKER=16
JOINT_ACC=0.0 # training all the tasks at once.
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr_randaug"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# iCaRL
FEAT_SIZE=2048

# BiC
distilling="--distilling" # Normal BiC. If you do not want to use distilling loss, then "".

if [ -d "tensorboard" ]; then
    rm -rf tensorboard
    echo "Remove the tensorboard dir"
fi

if [ "$DATASET" == "mnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="mlp400"
    N_EPOCH=5; BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar10" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=256; BATCHSIZE=128; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar100" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1
    MODEL_NAME="resnet32"
    N_EPOCH=256; BATCHSIZE=16; LR=0.03 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=100 N_CLS_A_TASK=20 N_TASKS=5
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=5
    fi

elif [ "$DATASET" == "imagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=1000 TOPK=5
    MODEL_NAME="resnet34"
    N_EPOCH=100; BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=10
    fi
else
    echo "Undefined setting"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode $MODE --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC --seed 2 \
    --seed 2 \
    --name cifar10-4K.2 \
    --expand-labels \
    --dataset cifar10 \
    --num-classes 10 \
    --num-labeled 4000 \
    --total-steps 5000 \
    --eval-step 100 \
    --randaug 2 16 \
    --batch-size 128 \
    --teacher_lr 0.05 \
    --student_lr 0.05 \
    --weight-decay 5e-4 \
    --ema 0.995 \
    --nesterov \
    --mu 7 \
    --label-smoothing 0.15 \
    --temperature 0.7 \
    --threshold 0.6 \
    --lambda-u 8 \
    --warmup-steps 500 \
    --uda-steps 500 \
    --student-wait-steps 300 \
    --teacher-dropout 0.2 \
    --student-dropout 0.2 \
    --finetune-epochs 30 \
    --finetune-batch-size 256 \
    --finetune-lr 3e-5 \
    --finetune-weight-decay 0 \
    --finetune-momentum 0.9 \
    --amp
