class LambdaScheduler:

    def __init__(self, warmup_epochs: int = 5, ramp_epochs: int = 10):
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def get_scale(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        elapsed = epoch - self.warmup_epochs
        return min(1.0, elapsed / self.ramp_epochs)
