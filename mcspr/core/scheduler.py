class LambdaScheduler:
    """Constant-lambda scheduler. Returns 1.0 at every epoch.

    Warmup/ramp removed per experimental finding that gradual ramp-in
    caused monotonic val_pcc_m degradation after full lambda engaged —
    consistent with the model building strong MSE-optimal weights over
    warmup epochs and being unable to accommodate the regulariser once
    it suddenly ramped in. Constant lambda from epoch 0 co-trains the
    correlation prior with the MSE signal from the first gradient step.

    Lambda magnitude is controlled entirely by lambda_max passed to
    MCSPRLoss (or via selected_lambda.json); this scheduler is purely
    a scale multiplier API.
    """

    def __init__(self, warmup_epochs: int = 0, ramp_epochs: int = 0):
        # Retained for API compatibility with callers that still pass these.
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def get_scale(self, epoch: int) -> float:
        return 1.0
