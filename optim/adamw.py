import math
import torch
from torch import optim


class AdamWBF16(optim.Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, cast_dtype=torch.float32,
                 is_normalize = False,
                 using_scaling = False,
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        self.using_scaling = using_scaling
        if self.using_scaling:
            print('using adamw scale')

        self.is_normalize = is_normalize
        if self.is_normalize:
            print('using norm grad')
        else:
            print('using non-norm grad')
        self.cast_dtype = cast_dtype
        super(AdamWBF16, self).__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        is_display = False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.to(dtype=self.cast_dtype)

                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                if state['step'] == 0:
                    if is_display == False:
                        print('dtype in adamw', grad.dtype)
                        is_display = True

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1


                if self.using_scaling:
                    # /Adapt Step from Adafactor
                    step_size = step_size * max(1e-3, self._rms(p.data))
                    # /Adapt Step from Adafactor

                p.grad.data.copy_( exp_avg/denom )

                if self.is_normalize:
                    g_norm = torch.mean(p.grad**2)**0.5
                    p.grad.div_(1e-30+g_norm)

                p.add_(p.grad, alpha=-step_size)

                #p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
