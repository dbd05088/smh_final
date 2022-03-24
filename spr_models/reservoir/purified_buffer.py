import numpy as np
from .prs import PRS, SubStream_Container
import random
import torch
import numpy as np
from collections import deque


class PurifiedBuffer(PRS):

    def reset(self):
        """reset the buffer.
        """
        self.rsvr = dict()
        self.rsvr_available_idx = deque(range(self.rsvr_total_size))
        self.substreams = SubStream_Container(self.rsvr_total_size)
        self.n = 0
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        return

    def sample_out(self):
        """
        select a sample and remove from rsvr
        """
        probs = self.substreams.get_probs()
        selected_key = np.random.choice(list(probs.keys()), 1, p=list(probs.values()))[0]
        # the least confident sample is sampled first.
        clean_weighting = []
        for idx in self.substreams[selected_key].idxs:
            clean_weighting.append(self.rsvr['clean_ps'][idx])

        z = self.substreams[selected_key].idxs[clean_weighting.index(min(clean_weighting))]
        self.remove_sample(z)

    def state(self, key):
        """override state function
        :param key: data names
        :return: states of reservoir
        """
        if key == "corrupts":
            ncorrupt = sum(self.rsvr[key]) - sum(self.rsvr[key][self.rsvr_available_idx])  # exclude examples in pool
            return "#normal data: {}, \t #corrupted data: {}".format(len(self) - ncorrupt, ncorrupt)

        return ""

