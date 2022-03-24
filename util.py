import logging
import os
import shutil
from collections import OrderedDict
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
import yaml
import torch
import torchvision.transforms as transforms
import numpy as np
import kornia
import scipy.stats as stats
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)

# -----------SPR util part start-----------
def setup_logger():
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

def override_config(config, override):
    # Override options
    for option in override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    return config

class SelfSupTransform():
    def __init__(self, image_shape):
        transform = [
            kornia.augmentation.RandomResizedCrop(size=image_shape[:2]),
            kornia.augmentation.RandomHorizontalFlip()]
        if image_shape[2] == 3:
            transform.append(kornia.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.2, p=0.5))
        self.transform = transforms.Compose(transform)
    def __call__(self, image):
        return self.transform(image)


class BetaMixture1D(object):
    """This code is based on the https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py"""
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    @staticmethod
    def fit_beta_weighted(x, w):
        def weighted_mean(x, w):
            return np.sum(w * x) / np.sum(w)

        x_bar = weighted_mean(x, w)
        s2 = weighted_mean((x - x_bar)**2, w)
        alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta = alpha * (1 - x_bar) /x_bar
        return alpha, beta

    @staticmethod
    def outlier_remove(x):
        # outliers detection
        # x에서 백분위 95%인 애와 5%인 애를 구해주는 것
        max_perc = np.percentile(x, 95)
        min_perc = np.percentile(x, 5)
        x = x[(x<=max_perc) & (x>=min_perc)]
        x_max = max_perc
        x_min = min_perc + 10e-6
        return x, x_min, x_max

    @staticmethod
    def normalize(x, x_min, x_max):
        # normalized the centrality for bmm
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x[x >= 1] =  1 -10e-4
        x[x <= 0] = 10e-4
        return x

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        # pi(mixing coefficient) * likelihood
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        # weighted likelihood를 모두 합한 값, 다시 말해서 분모
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        wl = self.weighted_likelihood(x, y)
        p = self.probability(x)

        # posterial을 구하는데 분모가 0이 되는 것을 방지하기 위해서 self.eps_nan을 더해준 것
        pos = wl / (p + self.eps_nan) 

        # infinity 값 있는지 판단
        wl_inf = np.isinf(wl)
        p_inf = np.isinf(p)

        # inf / inf -> 1
        pos[wl_inf & p_inf] = 1.
        return pos

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unstable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = self.fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = self.fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x = np.array((self.lookup_resolution * x).astype(int))
        x[x < 0] = 0
        x[x == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


class NTXentLoss(nn.Module):
    """This code is based on the https://github.com/chagmgang/simclr_pytorch/blob/master/nt_xent_loss.py"""
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

# -----------SPR util part end-----------

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def create_loss_fn(args):
    #if args.label_smoothing > 0:
    #    criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    #else:
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        if self.alpha == 0:
            loss = F.cross_entropy(logits, labels)
        else:
            num_classes = logits.shape[-1]
            alpha_div_k = self.alpha / num_classes
            target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                (1. - self.alpha) + alpha_div_k
            loss = (-(target_probs * torch.log_softmax(logits, dim=-1)
                      ).sum(dim=-1)).mean()
        return loss


class SmoothCrossEntropyV2(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, label_smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert label_smoothing < 1.0
        self.smoothing = label_smoothing
        self.confidence = 1. - label_smoothing

    def forward(self, x, target):
        if self.smoothing == 0:
            loss = F.cross_entropy(x, target)
        else:
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (self.confidence * nll_loss +
                    self.smoothing * smooth_loss).mean()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

