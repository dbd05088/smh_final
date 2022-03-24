import argparse
import logging
import math
import os
import random
import time
import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from spr_main import cls_SPR
import pandas as pd
import PIL
from tqdm import tqdm
from methods.finetune import Finetune

from model import WideResNet, ModelEMA
from util import (AverageMeter, accuracy, create_loss_fn,
                  save_checkpoint, reduce_tensor, model_load_state_dict)

transform_noisy = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor()])
class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None):
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)

        img_path = os.path.join("dataset", self.dataset, img_name)
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        sample["image_name"] = img_name
        return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]

class MetaPseudo:

    def get_dataloader(self, batch_size, noisy_batch_size, n_worker, labeled_list, unlabeled_list, test_list, noisy_list):
        # Loader
        labeled_loader = None
        unlabeled_loader = None
        test_loader = None
        if labeled_list is not None and len(labeled_list) > 0:
            #print("list len", len(labeled_list))
            labeled_dataset = ImageDataset(
                pd.DataFrame(labeled_list),
                dataset=self.dataset,
                transform=self.label_transform,
            )
            #print("dataset len", len(labeled_dataset))
            #print("batch size", batch_size)
            # drop last becasue of BatchNorm1D in IcarlNet
            labeled_loader = DataLoader(
                labeled_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )
            #print("loader len", len(labeled_loader))
        if unlabeled_list is not None and len(unlabeled_list) > 0:
            unlabeled_dataset = ImageDataset(
                pd.DataFrame(unlabeled_list),
                dataset=self.dataset,
                transform=self.unlabel_transform,
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                    test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        if noisy_list is not None and len(noisy_list) > 0:
            noisy_dataset = ImageDataset(
                pd.DataFrame(noisy_list),
                dataset=self.dataset,
                transform=transform_noisy,
            )
            data_sampler = SequentialSampler(noisy_dataset)
            noisy_loader = DataLoader(
                noisy_dataset,
                batch_size=noisy_batch_size,
                num_workers=n_worker,
                sampler = data_sampler,
                drop_last=True,
            )


        return labeled_loader, unlabeled_loader, test_loader, noisy_loader

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def get_cosine_schedule_with_warmup(self, optimizer,
                                        num_warmup_steps,
                                        num_training_steps,
                                        num_wait_steps=0,
                                        num_cycles=0.5,
                                        last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_wait_steps:
                return 0.0

            if current_step < num_warmup_steps + num_wait_steps:
                return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

            progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                float(max(1, num_training_steps -
                      num_warmup_steps - num_wait_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']

    def train_loop(self):
        if self.args.world_size > 1:
            labeled_epoch = 0
            unlabeled_epoch = 0
            self.labeled_loader.sampler.set_epoch(labeled_epoch)
            self.unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
        
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        # print(len(labeled_iter))

        # for author's code formula
        # moving_dot_product = torch.empty(1).to(self.args.device)
        # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
        # nn.init.uniform_(moving_dot_product, -limit, limit)

        # print("total_step :", self.args.total_steps)
        # print("eval_step :", self.args.eval_step)
        for step in range(self.args.start_step, self.args.total_steps):
            print("total :", self.args.total_steps, " current step : ", step)
            if step % self.args.eval_step == 0:
                pbar = tqdm(range(self.args.eval_step),
                            disable=self.args.local_rank not in [-1, 0])
                batch_time = AverageMeter()
                data_time = AverageMeter()
                s_losses = AverageMeter()
                t_losses = AverageMeter()
                t_losses_l = AverageMeter()  # labeled
                t_losses_u = AverageMeter()  # unlabeled
                t_losses_mpl = AverageMeter()
                mean_mask = AverageMeter()

            self.teacher_model.train()
            self.student_model.train()
            end = time.time()

            try:
                data_l = labeled_iter.next()
                images_l = data_l['image']
                targets = data_l['label']
                image_name_l = data_l['image_name']
                
            except:
                if self.args.world_size > 1:
                    labeled_epoch += 1
                    self.labeled_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(self.labeled_loader)
                data_l = labeled_iter.next()
                images_l = data_l['image']
                targets = data_l['label']
                image_name_l = data_l['image_name']

            try:
                data_ul = unlabeled_iter.next()
                images_ul = data_ul['image']
                targets_ul = data_ul['label']
                image_name_ul = data_ul['image_name']
                images_uw, images_us = images_ul[0], images_ul[1]
                
            except:
                if self.args.world_size > 1:
                    unlabeled_epoch += 1
                    self.unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(self.unlabeled_loader)
                # image_uw는 original, image_us는 augmentation 취한 것
                images_ul = data_ul['image']
                targets_ul = data_ul['label']
                image_name_ul = data_ul['image_name']
                images_uw, images_us = images_ul[0], images_ul[1]

            data_time.update(time.time() - end)

            # images_l과 targets는 from labeled_dataset
            images_l = images_l.to(self.args.device)
            images_uw = images_uw.to(self.args.device)
            images_us = images_us.to(self.args.device)
            targets = targets.to(self.args.device)
            with amp.autocast(enabled=self.args.amp):
                # batch size는 image의 크기 만큼!
                batch_size = images_l.shape[0]
                # tensor화
                t_images = torch.cat((images_l, images_uw, images_us))
                t_logits = self.teacher_model(t_images)
                t_logits_l = t_logits[:batch_size]
                # chunk(num)는 동일한 크기로 num만큼 쪼갠다.
                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                del t_logits

                # labeled의 loss
                t_loss_l = self.criterion(t_logits_l, targets)

                # dim=-1이라는 것은 softmax의 output 값의 마지막 차원을 제거한다는 의미이다.
                # input = k-dimensional vector이고 0 <= i <= k의 i번째 element에 대한 확률 값을 의미한다.
                soft_pseudo_label = torch.softmax(
                    t_logits_uw.detach() / self.args.temperature, dim=-1)

                # torch.max를 통해서 제일 큰 애를 return 해주는데, 만약 dim이 parameter로 같이 주어진다면
                # max인 index까지 return해주게 된다.
                # max_probs : 최대 확률 값들
                # hard_pseudo_label : 각 row별로 최대 확률 값들이 들어 있는 index들이 적힌 array
                # pseudo labeling을 한 것!
                max_probs, hard_pseudo_label = torch.max(
                    soft_pseudo_label, dim=-1)

                # psuedo label 결과
                self.unlabeled_data.append((images_uw, hard_pseudo_label))

                # max_probs안에 들어 있는 값이 threshold보다 큰 값이면 True, 아니면 false를 반환하도록
                # 만들어주는 operator가 ge(greater and equal) operator이다.
                # 이때 hard labeling을 하기 위해서 float()를 붙여주어 1.0 or 0.0으로 만들어준다.
                # 특정 threshold보다 커야만 인정하는 것
                # 구체적으로는 confidence based masking (확실한 Unlabeled data만을 이용하겠다는 뜻)
                mask = max_probs.ge(self.args.threshold).float()

                # element wise product
                # 앞에 -가 붙어 있는데, soft_pseudo_label과 t_logic_us의 softmax 값이 거의 같을수록 값이 작아질 것이다.
                # 따라서 그 sum 또한 작아질 것이고(음수), mask를 곱하여 의미 있는 값들을 다 합해주어 평균내주면 t_loss_u
                t_loss_u = torch.mean(
                    -(soft_pseudo_label *
                      torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                )
                weight_u = self.args.lambda_u * \
                    min(1., (step + 1) / self.args.uda_steps)
                t_loss_uda = t_loss_l + weight_u * t_loss_u

                s_images = torch.cat((images_l, images_us))
                s_logits = self.student_model(s_images)
                s_logits_l = s_logits[:batch_size]
                s_logits_us = s_logits[batch_size:]
                del s_logits

                # teacher model에서 생성한 pseudo model (x, y')을 바탕으로
                # student model에서 cross entropy loss를 고려
                # x, y'
                s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
                s_loss = self.criterion(s_logits_us, hard_pseudo_label)

            self.s_scaler.scale(s_loss).backward()
            if self.args.grad_clip > 0:
                self.s_scaler.unscale_(self.s_optimizer)
                nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), self.args.grad_clip)
            self.s_scaler.step(self.s_optimizer)
            self.s_scaler.update()
            self.s_scheduler.step()
            if self.args.ema > 0:
                self.avg_student_model.update_parameters(self.student_model)

            with amp.autocast(enabled=self.args.amp):
                with torch.no_grad():
                    s_logits_l = self.student_model(images_l)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
                # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
                # dot_product = s_loss_l_old - s_loss_l_new

                # author's code formula
                dot_product = s_loss_l_new - s_loss_l_old
                # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
                # dot_product = dot_product - moving_dot_product

                _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
                t_loss_mpl = dot_product * \
                    F.cross_entropy(t_logits_us, hard_pseudo_label)
                # test
                # t_loss_mpl = torch.tensor(0.).to(self.args.device)
                t_loss = t_loss_uda + t_loss_mpl

            self.t_scaler.scale(t_loss).backward()
            if self.args.grad_clip > 0:
                self.t_scaler.unscale_(self.t_optimizer)
                nn.utils.clip_grad_norm_(
                    self.teacher_model.parameters(), self.args.grad_clip)
            self.t_scaler.step(self.t_optimizer)
            self.t_scaler.update()
            self.t_scheduler.step()

            self.teacher_model.zero_grad()
            self.student_model.zero_grad()

            if self.args.world_size > 1:
                s_loss = reduce_tensor(s_loss.detach(), self.args.world_size)
                t_loss = reduce_tensor(t_loss.detach(), self.args.world_size)
                t_loss_l = reduce_tensor(
                    t_loss_l.detach(), self.args.world_size)
                t_loss_u = reduce_tensor(
                    t_loss_u.detach(), self.args.world_size)
                t_loss_mpl = reduce_tensor(
                    t_loss_mpl.detach(), self.args.world_size)
                mask = reduce_tensor(mask, self.args.world_size)

            s_losses.update(s_loss.item())
            t_losses.update(t_loss.item())
            t_losses_l.update(t_loss_l.item())
            t_losses_u.update(t_loss_u.item())
            t_losses_mpl.update(t_loss_mpl.item())
            mean_mask.update(mask.mean().item())

            batch_time.update(time.time() - end)
            pbar.set_description(
                f"Train Iter: {step+1:3}/{self.args.total_steps:3}. "
                f"LR: {self.get_lr(self.s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
                f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
            pbar.update()
            #if self.args.local_rank in [-1, 0]:
                #self.args.writer.add_scalar("lr", self.get_lr(self.s_optimizer), step)

            self.args.num_eval = step // self.args.eval_step
            if (step + 1) % self.args.eval_step == 0:
                pbar.close()
                if self.args.local_rank in [-1, 0]:
                    '''
                    self.args.writer.add_scalar(
                        "train/1.s_loss", s_losses.avg, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "train/2.t_loss", t_losses.avg, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "train/3.t_labeled", t_losses_l.avg, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "train/4.t_unlabeled", t_losses_u.avg, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "train/5.t_mpl", t_losses_mpl.avg, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "train/6.mask", mean_mask.avg, self.args.num_eval)
                    '''
                    test_model = self.avg_student_model if self.avg_student_model is not None else self.student_model
                    test_loss, top1, top5 = self.evaluate(test_model)
                    '''
                    self.args.writer.add_scalar(
                        "test/loss", test_loss, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "test/acc@1", top1, self.args.num_eval)
                    self.args.writer.add_scalar(
                        "test/acc@5", top5, self.args.num_eval)
                    '''
                    is_best = top1 > self.args.best_top1
                    if is_best:
                        self.args.best_top1 = top1
                        self.args.best_top5 = top5

                    save_checkpoint(self.args, {
                        'step': step + 1,
                        'teacher_state_dict': self.teacher_model.state_dict(),
                        'student_state_dict': self.student_model.state_dict(),
                        'avg_state_dict': self.avg_student_model.state_dict() if self.avg_student_model is not None else None,
                        'best_top1': self.args.best_top1,
                        'best_top5': self.args.best_top5,
                        'teacher_optimizer': self.t_optimizer.state_dict(),
                        'studenself.t_optimizer': self.s_optimizer.state_dict(),
                        'teacher_scheduler': self.t_scheduler.state_dict(),
                        'studenself.t_scheduler': self.s_scheduler.state_dict(),
                        'teacher_scaler': self.t_scaler.state_dict(),
                        'studenself.t_scaler': self.s_scaler.state_dict(),
                    }, is_best)

        #if self.args.local_rank in [-1, 0]:
            #self.args.writer.add_scalar("result/test_acc@1", self.args.best_top1)

        if self.args.finetune:
            del self.t_scaler, self.t_scheduler, self.t_optimizer, self.teacher_model, self.labeled_loader, self.unlabeled_loader
            del self.s_scaler, self.s_scheduler, self.s_optimizer

        if self.args.evaluate:
            del self.t_scaler, self.t_scheduler, self.t_optimizer, self.teacher_model, self.unlabeled_loader, self.labeled_loader
            del self.s_scaler, self.s_scheduler, self.s_optimizer
            evaluate(args, test_loader, student_model, criterion)
            return

        ckpt_name = f'{self.args.save_path}/{self.args.name}_best.pth.tar'
        loc = f'cuda:{self.args.gpu}'
        checkpoint = torch.load(ckpt_name, map_location=loc)
        if checkpoint['avg_state_dict'] is not None:
            model_load_state_dict(self.student_model,
                                  checkpoint['avg_state_dict'])
        else:
            model_load_state_dict(
                self.student_model, checkpoint['student_state_dict'])

        self.finetune(self.labeled_loader, self.student_model)

        # make hard pseudo label
        self.pseudo_loader = self.my_dataloader(128, 2, self.unlabel_list)
        pseudo_images = []
        pseudo_targets = []
        total = 0
        correct = 0
        with amp.autocast(enabled=self.args.amp):
            for idx, return_data in enumerate(self.pseudo_loader):
                images_uw = return_data['image']
                images_uw = images_uw.to(self.args.device)
                t_logits = self.teacher_model(images_uw)
                soft_pseudo_label = torch.softmax(
                    t_logits.detach() / self.args.temperature, dim=-1)
                #print("soft_pseudo_label")
                #print(soft_pseudo_label)
                max_probs, hard_pseudo_label = torch.max(
                    soft_pseudo_label, dim=-1)
                images_uw = images_uw.tolist()
                hard_pseudo_label = hard_pseudo_label.tolist()
                pseudo_images.extend(images_uw)
                pseudo_targets.extend(hard_pseudo_label)
                #print("hard_pseudo_label")
                #print(hard_pseudo_label)
                #print("original label")
                #print(return_data["label"])
                total += len(hard_pseudo_label)

                #print(torch.Tensor(hard_pseudo_label)==return_data["label"])
                # pseudo labeling 결과 측정
                c = torch.Tensor(hard_pseudo_label)==return_data["label"]
                compare = torch.Tensor.tolist(c)
                correct += compare.count(1)

        print("total", total, "correct", correct, "accuracy", (correct/total)*100)
        return pseudo_images, pseudo_targets

    def my_dataloader(self, batch_size, n_worker,  train_list):
        train_dataset = ImageDataset(
            pd.DataFrame(train_list),
            dataset=self.dataset,
            transform=self.train_transform,
        )
        # drop last becasue of BatchNorm1D in IcarlNet
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=16,
            drop_last=True,
        )
        return train_loader

    def evaluate(self, model):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.eval()
        test_iter = tqdm(self.test_loader,
                         disable=self.args.local_rank not in [-1, 0])
        with torch.no_grad():
            end = time.time()
            for step, tuples in enumerate(test_iter):
                images = tuples['image']
                targets = tuples['label']
                batch_size = images.shape[0]
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)
                with amp.autocast(enabled=self.args.amp):
                    outputs = model(images)
                    loss = self.criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, (1, 5))
                losses.update(loss.item(), batch_size)
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                batch_time.update(time.time() - end)
                end = time.time()
                test_iter.set_description(
                    f"Test Iter: {step+1:3}/{len(self.test_loader):3}. Data: {data_time.avg:.2f}s. "
                    f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

            test_iter.close()
            return losses.avg, top1.avg, top5.avg

    def finetune(self, finetune_loader, model):
        model.drop = nn.Identity()
        train_sampler = RandomSampler if self.args.local_rank == -1 else DistributedSampler

        optimizer = optim.SGD(model.parameters(),
                              lr=self.args.finetune_lr,
                              momentum=self.args.finetune_momentum,
                              weight_decay=self.args.finetune_weight_decay,
                              nesterov=True)
        scaler = amp.GradScaler(enabled=self.args.amp)

        for epoch in range(self.args.finetune_epochs):
            if self.args.world_size > 1:
                finetune_loader.sampler.set_epoch(epoch + 624)

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            model.train()
            end = time.time()
            labeled_iter = tqdm(
                finetune_loader, disable=self.args.local_rank not in [-1, 0])

            for step, datas in enumerate(labeled_iter):
                images = datas['image']
                targets = datas['label']
                data_time.update(time.time() - end)
                batch_size = images.shape[0]
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)
                with amp.autocast(enabled=self.args.amp):
                    model.zero_grad()
                    outputs = model(images)
                    loss = self.criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if self.args.world_size > 1:
                    loss = reduce_tensor(loss.detach(), self.args.world_size)
                losses.update(loss.item(), batch_size)
                batch_time.update(time.time() - end)
                labeled_iter.set_description(
                    f"Finetune Epoch: {epoch+1:2}/{self.args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                    f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
            labeled_iter.close()
            if self.args.local_rank in [-1, 0]:
                #self.args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
                test_loss, top1, top5 = self.evaluate(model)
                #self.args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
                #self.args.writer.add_scalar("finetune/acc@1", top1, epoch)
                #self.args.writer.add_scalar("finetune/acc@5", top5, epoch)

                is_best = top1 > self.args.best_top1
                if is_best:
                    self.args.best_top1 = top1
                    self.args.best_top5 = top5

                save_checkpoint(self.args, {
                    'step': step + 1,
                    'best_top1': self.args.best_top1,
                    'best_top5': self.args.best_top5,
                    'student_state_dict': model.state_dict(),
                    'avg_state_dict': None,
                    'studenself.t_optimizer': optimizer.state_dict(),
                }, is_best, finetune=True)
            #if self.args.local_rank in [-1, 0]:
                #self.args.writer.add_scalar("result/finetune_acc@1", self.args.best_top1)

        return

    def get_labeled_loader(self):
        return self.labeled_loader

    def __init__(self, method, args, labeled_list, unlabeled_list, test_list, noisy_list, transform_labeled, transform_unlabeled, transform_test, transform_train, origin):
        self.origin = origin
        self.train_transform = transform_train
        args.best_top1 = 0.
        args.best_top5 = 0.

        if args.local_rank != -1:
            args.gpu = args.local_rank
            torch.distributed.init_process_group(backend='nccl')
            args.world_size = torch.distributed.get_world_size()
        else:
            args.gpu = 0
            args.world_size = 1

        args.device = torch.device('cuda', args.gpu)

        self.dataset = args.dataset

        self.label_transform = transform_labeled
        self.unlabel_list = unlabeled_list
        self.unlabel_transform = transform_unlabeled
        self.test_transform = transform_test
        self.args = args

        #if self.args.local_rank in [-1, 0]:
            #self.args.writer = SummaryWriter(f"results/{self.args.name}")

        if self.args.seed is not None:
            self.set_seed()

        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        if self.args.local_rank == 0:
            torch.distributed.barrier()

        self.train_sampler = RandomSampler if self.args.local_rank == -1 else DistributedSampler
        print("*batch", self.args.batch_size)
        #print("*labeled len", len(labeled_list))
        self.labeled_loader, self.unlabeled_loader, self.test_loader, self.noisy_loader = self.get_dataloader(self.args.batch_size, 10, self.args.n_worker, labeled_list, unlabeled_list, test_list, noisy_list)

        print("-------noisy data-------")
        for idx, data in enumerate(self.noisy_loader):
            image_names = data['image_name']
            print("batch_size :", len(image_names))
            labels = data['label']
            corrupted = []
            for i in range(len(image_names)):
                corrupted.append(origin[image_names[i]]!=labels[i])
            data['corrupt'] = torch.Tensor(corrupted)
            break

        # 1. labeled_loader를 바탕으로 학습
        self.SPR = cls_SPR(self.args, self.noisy_loader, self.unlabeled_loader, self.origin)
        #print("noisy_loader_len", len(self.noisy_loader)) 80 (800 / 10) ...noisy_loader는 batch size가 10으로 지정되어 있음
        #print("unlabeled_loader_len", len(self.unlabeled_loader)) 78 (10000 / 128)
        #print("labeled_loader_len", len(self.labeled_loader)) 6 (800 / 128)
        self.SPR.spr_train()

        # 2. noisy_loader를 바탕으로 extract pure from noisy
        #print("-------pseudo check-------")
        #print(len(self.unlabeled_loader))
        #print("-------pseudo check-------")

        if self.args.dataset == "cifar10":
            depth, widen_factor = 28, 2
        elif self.args.dataset == "cifar100":
            depth, widen_factor = 28, 8

        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        self.teacher_model = WideResNet(num_classes=self.args.num_classes,
                                        depth=depth,
                                        widen_factor=widen_factor,
                                        dropout=0,
                                        dense_dropout=self.args.teacher_dropout)
        self.student_model = WideResNet(num_classes=self.args.num_classes,
                                        depth=depth,
                                        widen_factor=widen_factor,
                                        dropout=0,
                                        dense_dropout=self.args.student_dropout)

        if self.args.local_rank == 0:
            torch.distributed.barrier()

        self.teacher_model.to(self.args.device)
        self.student_model.to(self.args.device)
        self.avg_student_model = None
        if self.args.ema > 0:
            self.avg_student_model = ModelEMA(self.student_model, self.args.ema)

        self.criterion = create_loss_fn(self.args)

        no_decay = ['bn']
        teacher_parameters = [
            {'params': [p for n, p in self.teacher_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.teacher_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        student_parameters = [
            {'params': [p for n, p in self.student_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.student_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.t_optimizer = optim.SGD(teacher_parameters,
                                     lr=self.args.teacher_lr,
                                     momentum=self.args.momentum,
                                     nesterov=self.args.nesterov)
        self.s_optimizer = optim.SGD(student_parameters,
                                     lr=self.args.student_lr,
                                     momentum=self.args.momentum,
                                     nesterov=self.args.nesterov)

        self.t_scheduler = self.get_cosine_schedule_with_warmup(self.t_optimizer,
                                                                self.args.warmup_steps,
                                                                self.args.total_steps)
        self.s_scheduler = self.get_cosine_schedule_with_warmup(self.s_optimizer,
                                                                self.args.warmup_steps,
                                                                self.args.total_steps,
                                                                self.args.student_wait_steps)

        self.t_scaler = amp.GradScaler(enabled=args.amp)
        self.s_scaler = amp.GradScaler(enabled=args.amp)

        if self.args.resume:
            if os.path.isfile(self.args.resume):
                loc = f'cuda:{self.args.gpu}'
                checkpoint = torch.load(self.args.resume, map_location=loc)
                self.args.best_top1 = checkpoint['best_top1'].to(
                    torch.device('cpu'))
                self.args.best_top5 = checkpoint['best_top5'].to(
                    torch.device('cpu'))
                if not (self.args.evaluate or self.args.finetune):
                    self.args.start_step = checkpoint['step']
                    self.t_optimizer.load_state_dict(
                        checkpoint['teacher_optimizer'])
                    self.s_optimizer.load_state_dict(
                        checkpoint['studenself.t_optimizer'])
                    self.t_scheduler.load_state_dict(
                        checkpoint['teacher_scheduler'])
                    self.s_scheduler.load_state_dict(
                        checkpoint['studenself.t_scheduler'])
                    self.t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                    self.s_scaler.load_state_dict(
                        checkpoint['studenself.t_scaler'])
                    model_load_state_dict(
                        teacher_model, checkpoint['teacher_state_dict'])
                    if self.avg_student_model is not None:
                        model_load_state_dict(
                            self.avg_student_model, checkpoint['avg_state_dict'])

                else:
                    if checkpoint['avg_state_dict'] is not None:
                        model_load_state_dict(
                            self.student_model, checkpoint['avg_state_dict'])
                    else:
                        model_load_state_dict(
                            self.student_model, checkpoint['student_state_dict'])

        if self.args.local_rank != -1:
            teacher_model = nn.parallel.DistributedDataParallel(
                teacher_model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank, find_unused_parameters=True)
            student_model = nn.parallel.DistributedDataParallel(
                student_model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank, find_unused_parameters=True)

        if self.args.finetune:
            del self.t_scaler, self.t_scheduler, self.t_optimizer, self.teacher_model, self.unlabeled_loader
            del self.s_scaler, self.s_scheduler, self.s_optimizer
            # finetune 일단 labeled_num >= batch_size가 되는 경우가 없기 때문에 finetune_loader 자리에 labeled loader 넣음
            self.finetune(self.args, self.labeled_loader, self.test_loader,
                          self.student_model, self.criterion)
            return

        if self.args.evaluate:
            del self.t_scaler, self.t_scheduler, self.t_optimizer, self.teacher_model, self.unlabeled_loader, self.labeled_loader
            del self.s_scaler, self.s_scheduler, self.s_optimizer
            self.evaluate(student_model)
            return

        self.teacher_model.zero_grad()
        self.student_model.zero_grad()
        self.unlabeled_data = []
        '''
        train_loop(self.args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
                teacher_model, student_model, avg_student_model, criterion,
                self.t_optimizer, self.s_optimizer, self.t_scheduler, self.s_scheduler, self.t_scaler, self.s_scaler)
        '''
        return
