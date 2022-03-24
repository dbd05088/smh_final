"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            # yield는 generator를 return한다는 것을 제외하면 return과 다른 것이 없다.
            yield i


class RM(Finetune):
    def __init__(
        self, criterion, device, transform_labeled, transform_unlabeled, transform_test, n_classes, train_transform, test_transform, **kwargs
    ):
        super().__init__(
            criterion, device, transform_labeled, transform_unlabeled, transform_test, n_classes, train_transform, test_transform, **kwargs
        )
        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

    # self.streamed_list : online으로 들어오는 stream data
    # self.memory_list : 이전 task까지 sampling되어 memory 안에 들어있는 data들의 모음
    # self.train_list : len(train_list) + len(memory_list)
    # self.test_list : test samples

    # 중요!
    # 구조를 보면, task별로 train을 시키고 있다. 이 말은,
    # task를 다 훈련 시키고, memory 내부의 data를 재 학습 시키는 것이기 때문에,
    # task의 train data를 훈련시키는 동안에는 Catastrophic forgetting에 대해서 취약
    # task train이 다 끝나고 memory의 data까지 재학습 시켜야 완성되므로 이 부분이 취약 -> anytime reference가능하도록 바뀜
    # x = torch.cat([stream_data["image"], mem_data["image"]]) 이런 식으로 concat하기 때문!

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        # episodic memory로 저장되어 있는 data가 있는 것
        # then 기존 data를 memory loader에 넣어주어야 한다.

        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 3),
                num_workers=n_worker,
            )
            # stream data와 memory data의 batch size를 같도록
            pseudo_stream_batch_size = batch_size // 3
            stream_batch_size = batch_size - 2 * (batch_size // 3)
        else:
            memory_loader = None
            pseudo_stream_batch_size = batch_size // 2
            stream_batch_size = batch_size - pseudo_stream_batch_size

        print("batch_size :", batch_size, "stream_batch_size :", stream_batch_size, "memory_batch_size :", batch_size // 3)
        num_stuff = len(self.pseudo_images) // pseudo_stream_batch_size
        self.pseudo_loader = {}
        total_images = []
        total_labels = []
        base = 0
        for i in range(num_stuff):
            images = []
            labels = []
            for j in range(base, base + pseudo_stream_batch_size):
                images.append(self.pseudo_images[j])
                labels.append(self.pseudo_labels[j])

            images = torch.Tensor(images)
            labels = torch.Tensor(labels)

            total_images.append(images)
            total_labels.append(labels)
            # print("demo!!")
            # print(labels[:100])
            base += pseudo_stream_batch_size

        self.pseudo_loader['images'] = total_images
        self.pseudo_loader['labels'] = total_labels

        #print("@@@ batch size", batch_size)
        #print("@@@ pseudo batch size", pseudo_stream_batch_size)
        #print("@@@ stream batch size", stream_batch_size)
        #print('@@@ pseudo label len', len(self.pseudo_labels))

        # train_list == streamed_list in RM (RM은 online으로 training을 시키기 때문에 streamed_list를 바로 trian시킴)
        train_list = self.streamed_list
        test_list = self.test_list
        #random.shuffle(train_list)

        # batch with streamed and memory data equally하게 구성되어 있음
        # batch size를 stream_batch_size로 해준 이유는, 전체 batch size 중에서 나머지 batch size는 memory data로 채워질 것이기 때문이다.
        train_loader, test_loader = self.get_dataloader2(
            stream_batch_size, n_worker, train_list, test_list
        )
        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Pseudo samples: {len(self.pseudo_images)}")
        #logger.info(f"Per pseudo samples: {len(self.pseudo_images[0])}")
        #logger.info(f"shape image: {self.pseudo_images[0].shape}")

        # episodic memory에 저장되어 있는 data + stream으로 들어오는 data의 합
        logger.info(
            f"Train samples: {len(train_list)+len(self.memory_list)+len(self.pseudo_images)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)

        for epoch in range(n_epoch):
            # initialize for each task
            '''
            learning rate schedule이란?
            고정된 learning rate가 아닌, 미리 정해놓은 스케쥴대로 learning rate를 바꿔가면서 사용하는 것이다.
            특히 learning rate annealing은 learning rate가 iteration에 따라서 monotonically decreasing하는 경우를 의미한다.
            monotonically decreasing은 단조 감소 함수를 의미한다. (계속 감소하는)

            위와 같이 learning rate를 감소시키면, 초기에는 learning rate가 크기 때문에 빠르게 local minimum 부근으로 다가가고, 
            이후 learning rate를 줄여나가기 때문에 local minimum에 보다 더 정확하게 수렴하게 된다.

            warm restart란??
            학습 중간중간마다(ex. 1 epoch가 수행될 때 마다) learning rate를 증가시켜 주는 것
            만약 train dataset에 대한 loss function을 갖고, minimum weight 지점 w1을 찾았다고 해보자.
            하지만, test dataset에서는 train에 비해서 조금은 다른 loss function을 갖게 된다. 따라서 w1에서의 loss 값이 minimum이 아닐 수 있다.
            이때, 만약 가파른 지점 부근의 local minimum point였다면, 조금의 loss function change에서 실제 loss 값이 많이 차이나게 될 수 있다.
            따라서 이를 방지하기 위해서, 보다 완만한 local minimum, 즉 generalized된 곳을 찾아야 한다.
            이를 위해서 가끔씩 learning rate를 확 증가시켜, 가파른 local minimum에서 빠져나올 수 있는 기회를 제공하는 것을 warm restart라고 한다.
            '''
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go! (이게 우리가 알던 update 과정)
                self.scheduler.step()

            # memory data와 train data를 함께 train
            # 만약 episodic memory에 data가 들어 있다면 memory_loader is not none
            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            # tensor board 이용하는 부분
            print(f"task{cur_iter}/train/loss", train_loss, epoch)
            print(f"task{cur_iter}/test/loss",
                              eval_dict["avg_loss"], epoch)
            print(f"task{cur_iter}/test/acc",
                              eval_dict["avg_acc"], epoch)
            print(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)

        _, preds = logit.topk(self.topk, 1, True, True)

        loss.backward()
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(
        self, train_loader, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        # Sets the module in training mode.
        # cycle은 iterator를 사용할 때 쓰임
        # l = ['a', 'b', 'c']가 있을 때, next(l)을 계속 해주면 무한하게 a, b, c가 자동으로 반복됨
        self.model.train()
        if memory_loader is not None and train_loader is not None:
            data_iterator = zip(train_loader, cycle(memory_loader))
        elif memory_loader is not None:
            data_iterator = memory_loader
        elif train_loader is not None:
            data_iterator = train_loader
        else:
            raise NotImplementedError("None of dataloder is valid")

        for idx, data in enumerate(data_iterator):

            if len(data) == 2:
                # train_loader, memory_loader, pseudo_loader가 들어있는 경우
                stream_data, mem_data = data
                '''
                cat은 두 tensor를 concatenate해주는 함수
                tensor x1, x2의 차원이 둘다 (N, K)라고 하자
                torch.cat([x, y], dim=1) -> (N+N, K)
                torch.cat([x, y], dim=2) -> (N, K+K)
                '''
                # length check!!
                #print("---------length check-----------")
                #print(len(stream_data["image"]))
                #print(len(mem_data["image"]))
                #print(len(self.pseudo_loader['images']))
                #print("---------length check end-----------")

                stream_data["image"] = stream_data["image"].to(self.device)
                stream_data["label"] = stream_data["label"].type(torch.LongTensor)
                stream_data["label"] = stream_data["label"].to(self.device)

                self.pseudo_loader['images'][idx] = self.pseudo_loader['images'][idx].to(self.device)
                self.pseudo_loader['labels'][idx] = self.pseudo_loader['labels'][idx].type(torch.LongTensor)
                self.pseudo_loader['labels'][idx] = self.pseudo_loader['labels'][idx].to(self.device)

                mem_data["image"] = mem_data["image"].to(self.device)
                mem_data["label"] = mem_data["label"].type(torch.LongTensor)
                mem_data["label"] = mem_data["label"].to(self.device)
                x = torch.cat([stream_data["image"], mem_data["image"], self.pseudo_loader['images'][idx]])
                y = torch.cat([stream_data["label"], mem_data["label"], self.pseudo_loader['labels'][idx]])

            elif memory_loader is None:  # pseudo loader / stream loader data만 존재
                stream_data = data
                stream_data["image"] = stream_data["image"].to(self.device)
                stream_data["label"] = stream_data["label"].type(torch.LongTensor)
                stream_data["label"] = stream_data["label"].to(self.device)
                self.pseudo_loader['images'][idx] = self.pseudo_loader['images'][idx].to(self.device)
                self.pseudo_loader['labels'][idx] = self.pseudo_loader['labels'][idx].type(torch.LongTensor)
                self.pseudo_loader['labels'][idx] = self.pseudo_loader['labels'][idx].to(self.device)
                x = torch.cat([stream_data["image"], self.pseudo_loader['images'][idx]])
                y = torch.cat([stream_data["label"], self.pseudo_loader['labels'][idx]])

            else:
                x = data["image"]
                y = data["label"]

            y = y.type(torch.LongTensor)
            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y, criterion, optimizer)
            total_loss += l
            correct += c
            num_data += d

        if train_loader is not None:
            n_batches = len(train_loader)
        else:
            n_batches = len(memory_loader)

        #print("return_value(loss)", total_loss / n_batches)
        #print("return_value(correct)", correct / num_data)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size

