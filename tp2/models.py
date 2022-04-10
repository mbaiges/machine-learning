class Metrics():
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
    
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0

    def true_positive_rate(self):
        return self.tp / (self.tp + self.fn)

    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn)
    
    def f1_score(self):
        return 2 * self.precision() * self.true_positive_rate() / (self.precision() + self.true_positive_rate())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{" + f"acc: {self.accuracy():.3f}, prec: {self.precision():.3f}, tpr: {self.true_positive_rate():.3f}, fpr: {self.false_positive_rate():.3f}, f1s: {self.f1_score():.3f}" + "}"
