import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

def qr_positive_diag(A):
    Q, R = torch.linalg.qr(A)
    # signs of the diagonal of R
    signs = torch.sign(torch.diag(R))
    # handle exact zeros on diagonal (degenerate case)
    signs[signs == 0] = 1
    # flip columns of Q and rows of R
    Q = Q * signs.unsqueeze(0)   # broadcast over rows
    return Q


class KLOpt(torch.optim.Optimizer):
    """
    KLShampoo and KLSOAP

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.98), #depends on the batch size (for a large batch size, use beta2 = 0.95)
        shampoo_beta: float= -1,
        eps: float = 1e-9,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        using_klsoap: bool = False, #KL-Shampoo if False; KL-SOAP if True

        init_factor: float = 1e-3,

        using_clamping: bool = True, #using clamping in the eigenvalue estimation
        max_clamp_value: int = 4000,
        cast_dtype = torch.bfloat16, #use bfloat16 for all the computation (except the QR/eigen decomposition)
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
        }

        self.precondition_frequency = precondition_frequency
        self.cast_dtype = cast_dtype
        self.using_klsoap = using_klsoap
        class_name = 'kl-shampoo'
        if self.using_klsoap:
            class_name = 'kl-soap'

        self.using_clamping = using_clamping
        self.max_clamp_value = max_clamp_value
        if self.using_clamping:
            print('using eigenvalue clamping, max clamping value:', self.max_clamp_value)
        else:
            print('no clamping')

        self.init_factor = init_factor

        super().__init__(params, defaults)
        print('init factor:', self.init_factor)
        print('cast dtype:', self.cast_dtype)
        print('___________________________________%s public___________________________________'%class_name)


    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            state['GG'].append([])
        else:
            for sh in grad.shape:
                state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta


    @torch.compile
    def update_S(self, grad, state, mat, idx, beta, total_factor):
        factor = total_factor/grad.shape[idx]
        state['GG'][idx].mul_(beta).add_(mat, alpha=(1.0-beta)/factor)


    @torch.compile
    def update_eigen_value(self, state, diag, idx, beta):
        #update the eigen values of the curvature (every iteration)
        inv_d = state['eigen_sqrt_inv'][idx]**2
        D = torch.squeeze(1.0/inv_d).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        D.lerp_(diag, 1.0-beta)

        sqrt_inv_d = (1.0/torch.sqrt(D)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.using_clamping:
            sqrt_inv_d = torch.clamp(sqrt_inv_d, max=self.max_clamp_value) #####
        state['eigen_sqrt_inv'][idx] = sqrt_inv_d


    @torch.compile
    def update_2d_preconditioner(self, grad, state, total_factor):
        assert grad.dim() == 2 #only support 2d tensors
        beta = state["shampoo_beta"]

        for idx, sh in enumerate(grad.shape):
            o = state['Q'][abs(idx-1)]
            sqrt_inv_d = state['eigen_sqrt_inv'][abs(idx-1)]
            if idx == 0: #(left)
                step0 = o.T @ grad.T
                lhalf = step0 * sqrt_inv_d.view(-1,1)
                mat = lhalf.T @ lhalf
            else: #(right)
                step1 = o.T @ grad
                rhalf= step1 * sqrt_inv_d.view(-1,1)
                mat = rhalf.T @ rhalf

            self.update_S(grad, state, mat, idx, beta, total_factor)

        # diag_half = state['Q'][0].T @ grad @ state['Q'][1]
        diag_half = step1 @ state['Q'][1]
        ldiag = torch.mean( (diag_half * state['eigen_sqrt_inv'][1].view(1,-1))**2, 1)
        rdiag = torch.mean( (diag_half * state['eigen_sqrt_inv'][0].view(-1,1))**2, 0)

        self.update_eigen_value(state, ldiag, 1-1, beta)
        self.update_eigen_value(state, rdiag, 2-1, beta)


    @torch.no_grad()
    def update_preconditioner(self, grad, state):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        total_factor = torch.numel(grad)

        if state['Q'] is None:
            beta = state["shampoo_beta"]
            tmp = []
            for idx, sh in enumerate(grad.shape):
                mat = torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )
                tmp.append(mat)
                factor = total_factor/grad.shape[idx]
                state['GG'][idx] = mat/factor
                state['GG'][idx].diagonal().add_(1e-8) #better

            state['Q'], state['eigen_sqrt_inv'] = self.get_orthogonal_matrix(tmp)
            return                            # the statistic is diagonal in Q0: nothing to refresh at this step
        else:
            if self.using_klsoap and state['step'] % state['precondition_frequency'] == 0:
                state["exp_avg"] = self.project_back(state["exp_avg"], state)

            assert len(grad.shape)==2
            #only support 2d weights for now
            self.update_2d_preconditioner(grad, state, total_factor)

        if state['step'] < 10 or state['step'] % state['precondition_frequency'] == 0:
            #update the basis at the first 10 steps
            #update the basis of the curvature (every T iterations)
            state['Q'] = self.get_orthogonal_matrix_QR(state)
            if self.using_klsoap:
                state["exp_avg"] = self.project(state["exp_avg"], state)


    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = torch.squeeze( p.grad.to(dtype=self.cast_dtype) )
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    if self.using_klsoap:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                if 'Q' not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]),
                    )
                    self.update_preconditioner(grad, state)
                    continue # first step is skipped so that we never use the current gradients in the projection.

                state["step"] += 1

                if self.using_klsoap:
                    norm_grad = self.klsoap_update(state, grad, group["betas"][0], group["betas"][1], group['eps'])
                else:
                    norm_grad = self.klshampoo_update(state, grad, group["betas"][0], group['eps'])
                self.update_preconditioner(grad, state)

                step_size = group["lr"]

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                p.add_(norm_grad.view(p.shape), alpha=-step_size)

        return loss


    def klshampoo_update(self, state, grad, beta1, damping):
        exp_avg = state["exp_avg"]
        exp_avg.lerp_(grad, 1.0-beta1)

        grad_projected = self.project(exp_avg, state)
        #only support 2d weights now
        eigen_sqrt_inv = state['eigen_sqrt_inv'][0].view(-1,1) * state['eigen_sqrt_inv'][1].view(1,-1)

        eigen_sqrt_inv.div_( 1.0 + (eigen_sqrt_inv * damping) )
        precond_grad = grad_projected * eigen_sqrt_inv
        norm_grad = self.project_back(precond_grad, state)

        return norm_grad


    def klsoap_update(self, state, grad, beta1, beta2, damping):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        grad_projected = self.project(grad, state)
        exp_avg.lerp_(grad_projected, 1.0-beta1)
        exp_avg_sq.lerp_(grad_projected.square(), 1.0-beta2)

        denom = exp_avg_sq.sqrt().add_(damping)
        precond_grad = exp_avg / denom
        norm_grad = self.project_back(precond_grad, state)

        return norm_grad



    def project(self, grad, state):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [0]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return grad

    def project_back(self, grad, state):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [1]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return grad


    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        info = []
        assert len(matrix) == len(mat)
        for idx, m in enumerate(matrix):
            if len(m) == 0:
                final.append([])
                continue
            try:
                v0, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device))
            except:
                v0, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
                v0 = v0.to(m.dtype)

            Q = torch.flip(Q, [1])
            v = torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)*self.init_factor

            sqrt_inv_d = (1.0/torch.sqrt(v)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if not float_data:
                sqrt_inv_d = sqrt_inv_d.to(device=original_device, dtype=original_type)

            info.append(sqrt_inv_d)

            if not float_data:
                Q = Q.to(device=original_device, dtype=original_type)

            final.append(Q)
        return final, info

    def get_orthogonal_matrix_QR(self, state):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state['GG']
        orth_list = state['Q']

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
            assert len(m) > 0
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            power_iter = m @ o
            Q = qr_positive_diag(power_iter)

            if not float_data:
                Q = Q.to(device=original_device, dtype=original_type)

            final.append(Q)
        return final
