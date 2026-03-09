import os
import numpy as np
import random
from itertools import combinations
from collections import Counter


class KSampler:

    def __init__(self, args, n_items, l2 = 1e-5, K = 2, max_tuples = 200000):
        assert K <= 2, "Only K=1 or K=2 supported currently."

        self.args = args
        self.n = n_items
        self.eps = args.eps
        self.alpha = args.alpha
        self.l2 = l2
        self.K = K
        self.w = np.zeros(self.n)
        self.W2 = [{} for _ in range(self.n)]
        self.global_avg = 0.5
        self.max_tuples = max_tuples
        self.count = 0
        self.history = []

        self.n_updates = 0

    def _gain(self, S_mask):
        g = self.w.copy()
        idx = np.where(S_mask)[0]
        for j in idx:
            for i, v in self.W2[j].items():
                g[i] += v

        g[S_mask] = -np.inf
        return g

    def sample(self, k, explore = True):
        eps = self.eps
        if not explore:
            eps = 0.0

        S_mask = np.zeros(self.n, bool)
        chosen = []
        for _ in range(min(k, self.n)):
            g = self._gain(S_mask)
            mask = ~S_mask
            p = np.exp(g - np.nanmax(g[mask]))
            p[~mask] = 0
            s = p.sum()
            if s == 0: p = mask.astype(float) / mask.sum()
            else: p = p / s
            p = (1 - eps) * p + eps * (mask.astype(float) / mask.sum())
            i = np.random.choice(self.n, p = p)
            chosen.append(i)
            S_mask[i] = True
        return chosen

    def _decay(self):
        self.w *= (1 - self.l2)
        for j in range(self.n):
            for i in list(self.W2[j].keys()):
                self.W2[j][i] *= (1 - self.l2)

    def update(self, batch, score, subsamples = 64):
        b = np.asarray(batch)
        k = len(b)
        if k == 0: return
        credit = float(score) - self.global_avg
        self._decay()

        # item credit
        self.w[b] = (1 - self.alpha) * self.w[b] + self.alpha * (credit / k)

        # sample a few subsets for pair/triad credit
        for _ in range(subsamples):
            r = np.random.choice(b, size = min(3, k), replace = False)
            if len(r) >= 2:
                a, b2 = sorted(r[:2])
                val = self.W2[a].get(b2, 0.0)
                self.W2[a][b2] = (1 - self.alpha) * val + self.alpha * (credit / 2)
                self.W2[b2][a] = self.W2[a][b2]

        self.global_avg = 0.95 * self.global_avg + 0.05 * float(score)

    def _marginals(self, S_mask):
        """Return marginal gain g(i | S) for all i, using learned w, W2, W3."""
        g = self.w.copy()
        idx = np.where(S_mask)[0]

        # pair contributions
        for j in idx:
            for i, v in self.W2[j].items():
                g[i] += v

        g[S_mask] = -np.inf  # already chosen
        return g

    def estimate_set_score(self, S):
        """Offline estimate of batch score using learned parameters."""
        S = np.asarray(S, dtype = int)
        s = self.global_avg + self.w[S].sum()

        # pairs
        if len(S) >= 2:
            for a, b in combinations(S, 2):
                if b in self.W2[a]:
                    s += self.W2[a][b]

        return float(s)

    def update_many(self, list_of_idx_score):
        random.shuffle(list_of_idx_score)
        for idxs, score in list_of_idx_score:
            self.update(idxs, score)

        self.n_updates += 1

    def best_batch(self, k):
        """Greedy MAP: build size-k set by repeatedly adding max marginal."""
        k = int(min(k, self.n))
        S_mask = np.zeros(self.n, dtype = bool)
        chosen = []
        for _ in range(k):
            g = self._marginals(S_mask)
            i = int(np.nanargmax(g))
            if not np.isfinite(g[i]):
                break
            chosen.append(i)
            S_mask[i] = True
        return chosen, self.estimate_set_score(chosen)

    def save(self, path = None):
        if path is None:
            path = self.args.sampler_checkpoint_path

        os.makedirs(path, exist_ok = True)
        np.savez_compressed(path + 'sampler.npz', w = self.w, W2 = self.W2, global_avg = self.global_avg, allow_pickle = True, history = self.history)

    def load(self, path):
        if not os.path.exists(path + 'sampler.npz'):
            print(":: No saved sampler found, initializing new one.")
            return None

        data = np.load(path + 'sampler.npz', allow_pickle = True)

        self.w = data['w']
        self.W2 = data['W2']
        self.global_avg = float(data['global_avg'])
        self.history = data['history'].tolist()
        return self


if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(eps = 0.1, alpha = 0.99)
    sampler = KSampler(args, n_items = 50, K = 1)

    def get_score(batch_idxs):
        score = 0
        if 4 in batch_idxs:
            score += 1

        if 8 in batch_idxs:
            score += 1

        if 3 in batch_idxs and 4 in batch_idxs:
            score += 2

        if 4 in batch_idxs and 13 in batch_idxs and 3 in batch_idxs:
            score += 3

        return score

    counter = Counter()
    for rounds in range(10):
        round_indices = []
        round_scores = []

        for i in range(10):
            batch_idxs = [sampler.sample(4) for _ in range(10)]
            batch_scores = [get_score(b) for b in batch_idxs]

            for b, s in zip(batch_idxs, batch_scores):
                round_indices.append(b)
                round_scores.append(s)

        sampler.update_many(list(zip(round_indices, round_scores)))
        for b in round_indices:
            counter.update([tuple(sorted(b))])

        print(counter.most_common(8))

    print("Best batch:", sampler.best_batch(4))
    print("Best batch:", sampler.sample(4))
