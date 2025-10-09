import math

class OnlineStats:
    __slots__ = ("n","mean","M2","min","max")

    def __init__(self):
        self.n=0; self.mean=0.0; self.M2=0.0; self.min=float("inf"); self.max=float("-inf")

    def update(self, x: float):
        self.n += 1
        if x < self.min: self.min = x
        if x > self.max: self.max = x
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
        
    def stats(self):  # (min, mean, max, std)
        var = self.M2/(self.n-1) if self.n>1 else 0.0
        std = math.sqrt(var)
        mmin = 0.0 if self.n==0 or self.min==float("inf") else self.min
        mmax = 0.0 if self.n==0 or self.max==float("-inf") else self.max
        mmean= 0.0 if self.n==0 else self.mean
        return mmin, mmean, mmax, std

class EntropyCounter:
    __slots__ = ("bins","taken","K")

    def __init__(self, K):
        self.bins=[0]*256; self.taken=0; self.K=K

    def ingest(self, payload: bytes):
        if self.K <= 0 or self.taken >= self.K or not payload: return
        room = self.K - self.taken
        chunk = payload[:room]
        for b in chunk: self.bins[b]+=1
        self.taken += len(chunk)

    def entropy(self):
        if self.taken==0: return 0.0
        H=0.0; inv=1.0/self.taken
        for c in self.bins:
            if c:
                p=c*inv; H -= p*math.log2(p)
        return H

    def nonzero_frac(self):
        if self.taken==0: return 0.0
        return 1.0 - (self.bins[0]/self.taken)