import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

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
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.98),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int=10,
        using_klsoap: bool = False, #KL-Shampoo if False; KL-SOAP if True

        normalize_grads: bool = False,
        init_factor: float = 0.1,
        using_damping: bool = False,

        using_clamping: bool = True,
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
            "normalize_grads": normalize_grads,
        }
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
        self.using_damping = using_damping
        if using_damping:
            print('using damping in the curvature learning')
            self.damping = eps
        else:
            print('no damping in the curvature learning')
            self.damping = 0.0

        super().__init__(params, defaults)
        print('init factor:', self.init_factor)
        print('cast dtype:', self.cast_dtype)
        print('___________________________________%s org___________________________________'%class_name)


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
    def update_eigen_value(self, state, diag, idx, beta, traces, total_trace, damping):
        if damping > 0:
            diag = diag + total_trace/traces[idx]

        #update the eigen values of the curvature (every iteration)
        inv_d = state['eigen_sqrt_inv'][idx]**2
        D = torch.squeeze(1.0/inv_d).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        D.lerp_(diag, 1.0-beta)
        sqrt_inv_d = (1.0/torch.sqrt(D)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.using_clamping:
            sqrt_inv_d = torch.clamp(sqrt_inv_d, max=max(10, min(D.shape[0], self.max_clamp_value))) #clamp the inv_sqrt_eigen_value
        state['eigen_sqrt_inv'][idx] = sqrt_inv_d


    @torch.compile
    def update_3d_preconditioner(self, grad, state, total_factor, traces, total_trace,  damping):
        assert grad.dim() == 3 #only support 3d tensors
        invS_half = []
        for idx in range(grad.dim()):
            invS_h = state['Q'][idx] * state['eigen_sqrt_inv'][idx].view(1,-1)
            invS_half.append(invS_h)

        G = grad
        beta = state["shampoo_beta"]

        GinvS1h = torch.einsum('ija,ip->pja', G, invS_half[1-1]) #G invS1
        GinvS1Q2 = torch.einsum('pja,jl->pla', GinvS1h, state['Q'][2-1]) #G invS1 Q2
        GinvS1Q2Q3 = torch.einsum('pqa,am->pqm', GinvS1Q2, state['Q'][3-1])#G invS1 Q2 Q3

        GinvS12h = GinvS1Q2 * state['eigen_sqrt_inv'][2-1].view(1,-1,1) #G invS1 invS2
        GinvS12G_T = torch.tensordot(GinvS12h,GinvS12h, dims=[[0,1],[0,1]]) #for S3
        self.update_S(G, state, GinvS12G_T, 3-1, beta, total_factor) #update S3

        GinvS1Q3 = torch.einsum('pqb,bm->pqm', GinvS1h, state['Q'][3-1]) #G invS1 Q3
        GinvS13h = GinvS1Q3 * state['eigen_sqrt_inv'][3-1].view(1,1,-1) #G invS1 invS3
        GinvS13G_T = torch.tensordot(GinvS13h,GinvS13h, dims=[[0,2],[0,2]]) #for S2
        self.update_S(G, state, GinvS13G_T, 2-1, beta, total_factor) #update S2

        diag3 = torch.mean( (GinvS1Q2Q3*state['eigen_sqrt_inv'][2-1].view(1,-1,1))**2, dim=(0,1) ) #torch.diag( Q3.T @ GinvS12G_T @ Q3 )/(d1*d2) #for diag(S3)
        self.update_eigen_value(state, diag3, 3-1, beta, traces, total_trace, damping)

        diag2 = torch.mean( (GinvS1Q2Q3*state['eigen_sqrt_inv'][3-1].view(1,1,-1))**2, dim=(0,2) ) #torch.diag( Q2.T @ GinvS13G_T @ Q2 )/(d1*d3) #for diag(S2)
        self.update_eigen_value(state, diag2, 2-1, beta, traces, total_trace, damping)


        GinvS3h = torch.einsum('ijb,bm->ijm', G, invS_half[3-1]) #G invS3
        GinvS3Q2 = torch.einsum('ijm,jq->iqm', GinvS3h, state['Q'][2-1])
        GinvS3Q2Q1 = torch.einsum('iqm,ip->pqm', GinvS3Q2, state['Q'][1-1]) #G invS3 Q2 Q1

        GinvS32h = GinvS3Q2 * state['eigen_sqrt_inv'][2-1].view(1,-1,1) #G invS3 invS2
        GinvS32G_T = torch.tensordot(GinvS32h,GinvS32h, dims=[[1,2],[1,2]]) #for S1
        self.update_S(G, state, GinvS32G_T, 1-1, beta, total_factor) #update S1

        diag1 = torch.mean( (GinvS3Q2Q1*state['eigen_sqrt_inv'][2-1].view(1,-1,1))**2, dim=(1,2) ) #torch.diag( Q1.T @ GinvS32G_T @ Q1 )/(d2*d3) #for diag(S1)
        self.update_eigen_value(state, diag1, 1-1, beta, traces, total_trace, damping)


    @torch.compile
    def update_2d_preconditioner(self, grad, state, total_factor, traces, total_trace, damping):
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

        self.update_eigen_value(state, ldiag, 1-1, beta, traces, total_trace, damping)
        self.update_eigen_value(state, rdiag, 2-1, beta, traces, total_trace, damping)


    @torch.no_grad()
    def update_preconditioner(self, grad, state):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        traces = []
        total_factor = torch.numel(grad)
        damping = 0.0 #do not add damping in the curvature learning
        if self.using_damping:
            damping = self.damping
        total_trace = damping

        if damping > 0:
            for idx, sh in enumerate(grad.shape):
                if state['Q'] is None:
                    cur_trace = 1.0 #average
                else:
                    cur_trace = torch.mean(state['eigen_sqrt_inv'][idx]**2) #average
                total_trace *= cur_trace
                traces.append(cur_trace)

        if state['Q'] is None:
            beta = state["shampoo_beta"]
            for idx, sh in enumerate(grad.shape):
                mat = torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )
                self.update_S(grad, state, mat, idx, beta, total_factor)

            state['Q'], state['eigen_sqrt_inv'] = self.get_orthogonal_matrix(state['GG'])
        else:
            if self.using_klsoap and state['step'] % state['precondition_frequency'] == 0:
                state["exp_avg"] = self.project_back(state["exp_avg"], state)

            if len(grad.shape)==2:
                self.update_2d_preconditioner(grad, state, total_factor, traces, total_trace,  damping)
            elif len(grad.shape)==3:
                self.update_3d_preconditioner(grad, state, total_factor, traces, total_trace,  damping)
            else:
                #only support up to 3d tensors for now
                assert False

        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            #update the eigen bases of the curvature (every T iterations)
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

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)
                
                step_size = group["lr"]
                p.add_(norm_grad.view(p.shape), alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    
        return loss

    def klshampoo_update(self, state, grad, beta1, damping):
        exp_avg = state["exp_avg"]
        exp_avg.lerp_(grad, 1.0-beta1)

        grad_projected = self.project(exp_avg, state)
        if len(grad.shape) == 2:
            eigen_sqrt_inv = state['eigen_sqrt_inv'][0].view(-1,1) * state['eigen_sqrt_inv'][1].view(1,-1)
        elif len(grad.shape) == 3:
            eigen_sqrt_inv = state['eigen_sqrt_inv'][0].view(-1,1,1) * state['eigen_sqrt_inv'][1].view(1,-1,1) * state['eigen_sqrt_inv'][2].view(1,1,-1)
        else:
            assert False

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
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(device=original_device, dtype=original_type)
            final.append(Q)
        return final
