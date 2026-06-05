import torch
from torch import nn
import torch.nn.functional as F


class MoR_Linear(nn.Module):
    def __init__(self, raw_linear: nn.Linear, r, alpha, num_loras):
        super().__init__()

        in_features = raw_linear.in_features
        out_features = raw_linear.out_features
        bias = raw_linear.bias is not None

        if not isinstance(r, (list, tuple)):
            r = [r for i in range(num_loras)]
        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha for i in range(num_loras)]
        assert len(r) == len(alpha) == num_loras, (r, alpha)

        self.lora_as = nn.ModuleList(nn.Linear(in_features=in_features, out_features=r[i], bias=False)
                                     for i in range(len(r)))
        self.lora_bs = nn.ModuleList(nn.Linear(in_features=r[i], out_features=out_features, bias=False)
                                     for i in range(len(r)))

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        self.r = r
        self.alpha = alpha

        # +1 for the bypass / raw-weight column at index 0
        self.route = nn.Linear(in_features=in_features, out_features=num_loras + 1)

        self.init_lora()
        self.load_weight(raw_linear=raw_linear)
        
    def init_lora(self):
        for i in range(len(self.lora_as)):
            nn.init.kaiming_uniform_(self.lora_as[i].weight, a=2.236)  # sqrt(5)=2.236
            nn.init.zeros_(self.lora_bs[i].weight)
            # nn.init.kaiming_uniform_(self.lora_bs[i].weight)
        
    def load_weight(self, raw_linear):
        with torch.no_grad():
            self.linear.weight.copy_(raw_linear.weight)
            if self.linear.bias is not None:
                self.linear.bias.copy_(raw_linear.bias)
                
    def forward(self, x):
        # x: B, N, Din
        out = self.linear(x)  # B, N, Dout

        dxs = []
        for i in range(len(self.r)):
            dx = (self.alpha[i] / self.r[i]) * self.lora_bs[i](self.lora_as[i](x))  # B, N, Dout
            dxs.append(dx)

        expert_weights = F.softmax(self.route(x), dim=-1)
        expert_weights = expert_weights[:, :, 1:]  # drop bypass column → B, N, num_loras
        expert_weights = expert_weights.unsqueeze(3)  # B, N, num_loras, 1

        dxs = torch.stack(dxs, dim=2)  # B, N, num_loras, Dout
        dxs = torch.sum(expert_weights * dxs, dim=2)
        out += dxs
        return out


class MoR_qkv(nn.Module):
    def __init__(self, raw_qkv_linear: nn.Linear, r, alpha, num_loras, lora_q, lora_k, lora_v):
        super().__init__()

        dim = raw_qkv_linear.in_features
        self.dim = dim
        assert raw_qkv_linear.out_features == 3 * dim, (raw_qkv_linear)
        bias = raw_qkv_linear.bias is not None

        if not isinstance(r, (list, tuple)):
            r = [r for i in range(num_loras)]
        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha for i in range(num_loras)]
        assert len(r) == len(alpha) == num_loras, (r, alpha)

        self.linear = nn.Linear(in_features=dim, out_features=dim*3, bias=bias)

        self.lora_q = lora_q
        self.lora_k = lora_k
        self.lora_v = lora_v
        assert lora_q or lora_k or lora_v  # at least one lora
        if self.lora_q:
            self.lora_as_q = nn.ModuleList(nn.Linear(in_features=dim, out_features=r[i], bias=False)
                                         for i in range(num_loras))
            self.lora_bs_q = nn.ModuleList(nn.Linear(in_features=r[i], out_features=dim, bias=False)
                                         for i in range(num_loras))
            self.route_q = nn.Linear(in_features=dim, out_features=num_loras + 1)
        if self.lora_k:
            self.lora_as_k = nn.ModuleList(nn.Linear(in_features=dim, out_features=r[i], bias=False)
                                         for i in range(num_loras))
            self.lora_bs_k = nn.ModuleList(nn.Linear(in_features=r[i], out_features=dim, bias=False)
                                         for i in range(num_loras))
            self.route_k = nn.Linear(in_features=dim, out_features=num_loras + 1)
        if self.lora_v:
            self.lora_as_v = nn.ModuleList(nn.Linear(in_features=dim, out_features=r[i], bias=False)
                                         for i in range(num_loras))
            self.lora_bs_v = nn.ModuleList(nn.Linear(in_features=r[i], out_features=dim, bias=False)
                                         for i in range(num_loras))
            self.route_v = nn.Linear(in_features=dim, out_features=num_loras + 1)

        self.r = r
        self.alpha = alpha

        self.init_lora()
        self.load_weight(raw_linear=raw_qkv_linear)
        
    def init_lora(self):
        if self.lora_q:
            for i in range(len(self.lora_as_q)):
                nn.init.kaiming_uniform_(self.lora_as_q[i].weight, a=2.236)  # a=sqrt(5)
                nn.init.zeros_(self.lora_bs_q[i].weight)
                #nn.init.kaiming_uniform_(self.lora_bs_q[i].weight)
        if self.lora_k:
            for i in range(len(self.lora_as_k)):
                nn.init.kaiming_uniform_(self.lora_as_k[i].weight, a=2.236)  # a=sqrt(5)
                nn.init.zeros_(self.lora_bs_k[i].weight)
                #nn.init.kaiming_uniform_(self.lora_bs_k[i].weight)
        if self.lora_v:
            for i in range(len(self.lora_as_v)):
                nn.init.kaiming_uniform_(self.lora_as_v[i].weight, a=2.236)  # a=sqrt(5)
                nn.init.zeros_(self.lora_bs_v[i].weight)
                #nn.init.kaiming_uniform_(self.lora_bs_v[i].weight)

    def load_weight(self, raw_linear):
        with torch.no_grad():
            self.linear.weight.copy_(raw_linear.weight)
            if self.linear.bias is not None:
                self.linear.bias.copy_(raw_linear.bias)
                
    def forward(self, x):
        qkv = self.linear(x)  # B, N, 3*dim
        if self.lora_q:
            dqs = []
            for i in range(len(self.r)):
                dq = (self.alpha[i] / self.r[i]) * self.lora_bs_q[i](self.lora_as_q[i](x))  # B, N, Dout
                dqs.append(dq)
            dqs = torch.stack(dqs, dim=2)  # B, N, num_loras, Dout

            expert_weights = F.softmax(self.route_q(x), dim=-1)
            expert_weights = expert_weights[:, :, 1:]
            expert_weights = expert_weights.unsqueeze(3)

            dqs = torch.sum(expert_weights * dqs, dim=2)
            qkv[:, :, :self.dim] += dqs

        if self.lora_k:
            dks = []
            for i in range(len(self.r)):
                dk = (self.alpha[i] / self.r[i]) * self.lora_bs_k[i](self.lora_as_k[i](x))
                dks.append(dk)
            dks = torch.stack(dks, dim=2)

            expert_weights = F.softmax(self.route_k(x), dim=-1)
            expert_weights = expert_weights[:, :, 1:]
            expert_weights = expert_weights.unsqueeze(3)

            dks = torch.sum(expert_weights * dks, dim=2)
            qkv[:, :, self.dim:2*self.dim] += dks

        if self.lora_v:
            dvs = []
            for i in range(len(self.r)):
                dv = (self.alpha[i] / self.r[i]) * self.lora_bs_v[i](self.lora_as_v[i](x))
                dvs.append(dv)
            dvs = torch.stack(dvs, dim=2)

            expert_weights = F.softmax(self.route_v(x), dim=-1)
            expert_weights = expert_weights[:, :, 1:]
            expert_weights = expert_weights.unsqueeze(3)

            dvs = torch.sum(expert_weights * dvs, dim=2)
            qkv[:, :, 2*self.dim:] += dvs
        return qkv
    

def set_openclip_mor(model, r, alpha, lora_layers, num_loras):
    # lora_layers: q, k, v, proj, mlp
    if 'q' in lora_layers:
        lora_q = True
    else:
        lora_q = False
        
    if 'k' in lora_layers:
        lora_k = True
    else:
        lora_k = False
        
    if 'v' in lora_layers:
        lora_v = True
    else:
        lora_v = False
        
    if 'proj' in lora_layers:
        lora_proj = True
    else:
        lora_proj = False
        
    if 'mlp_fc1' in lora_layers or 'mlp_fc2' in lora_layers:
        lora_mlp = True
    else:
        lora_mlp = False
        
    print('lora_q: {}, lora_k: {}, lora_v: {}, lora_proj: {}, lora_proj: {}, lora_mlp: {}'.format(
        lora_q, lora_k, lora_v, lora_proj, lora_proj, lora_mlp))
    
    for i in range(len(model.blocks)):
        if lora_q or lora_k or lora_v:
            model.blocks[i].attn.qkv = MoR_qkv(raw_qkv_linear=model.blocks[i].attn.qkv,
                                                   lora_q=lora_q, lora_k=lora_k, lora_v=lora_v,
                                                   r=r, alpha=alpha, num_loras=num_loras)
        if lora_proj:
            model.blocks[i].attn.proj = MoR_Linear(raw_linear=model.blocks[i].attn.proj,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            
        if lora_mlp:
            if 'mlp_fc1' in lora_layers:
                model.blocks[i].mlp.fc1 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc1,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            if 'mlp_fc2' in lora_layers:
                model.blocks[i].mlp.fc2 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc2,
                                                       r=r, alpha=alpha, num_loras=num_loras)
    

def set_dinov2_mor(model, r, alpha, lora_layers, num_loras):
    # lora_layers: q, k, v, proj, mlp
    if 'q' in lora_layers:
        lora_q = True
    else:
        lora_q = False
        
    if 'k' in lora_layers:
        lora_k = True
    else:
        lora_k = False
        
    if 'v' in lora_layers:
        lora_v = True
    else:
        lora_v = False
        
    if 'proj' in lora_layers:
        lora_proj = True
    else:
        lora_proj = False
        
    if 'mlp_fc1' in lora_layers or 'mlp_fc2' in lora_layers:
        lora_mlp = True
    else:
        lora_mlp = False
        
    print('lora_q: {}, lora_k: {}, lora_v: {}, lora_proj: {}, lora_proj: {}, lora_mlp: {}'.format(
        lora_q, lora_k, lora_v, lora_proj, lora_proj, lora_mlp))
    
    for i in range(len(model.blocks)):
        if lora_q or lora_k or lora_v:
            model.blocks[i].attn.qkv = MoR_qkv(raw_qkv_linear=model.blocks[i].attn.qkv, 
                                                   lora_q=lora_q, lora_k=lora_k, lora_v=lora_v,
                                                   r=r, alpha=alpha, num_loras=num_loras
                                                )
        
        if lora_proj:
            model.blocks[i].attn.proj = MoR_Linear(raw_linear=model.blocks[i].attn.proj,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            
        if lora_mlp:
            if 'mlp_fc1' in lora_layers:
                model.blocks[i].mlp.fc1 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc1,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            if 'mlp_fc2' in lora_layers:
                model.blocks[i].mlp.fc2 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc2,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            


def set_radio_dino_mor(model, r, alpha, lora_layers, num_loras):
    # lora_layers: q, k, v, proj, mlp
    if 'q' in lora_layers:
        lora_q = True
    else:
        lora_q = False
        
    if 'k' in lora_layers:
        lora_k = True
    else:
        lora_k = False
        
    if 'v' in lora_layers:
        lora_v = True
    else:
        lora_v = False
        
    if 'proj' in lora_layers:
        lora_proj = True
    else:
        lora_proj = False
        
    if 'mlp_fc1' in lora_layers or 'mlp_fc2' in lora_layers:
        lora_mlp = True
    else:
        lora_mlp = False
        
    print('lora_q: {}, lora_k: {}, lora_v: {}, lora_proj: {}, lora_proj: {}, lora_mlp: {}'.format(
        lora_q, lora_k, lora_v, lora_proj, lora_proj, lora_mlp))
    
    for i in range(len(model.blocks)):
        if lora_q or lora_k or lora_v:
            model.blocks[i].attn.qkv = MoR_qkv(raw_qkv_linear=model.blocks[i].attn.qkv, 
                                                   lora_q=lora_q, lora_k=lora_k, lora_v=lora_v,
                                                   r=r, alpha=alpha, num_loras=num_loras
                                                )
        
        if lora_proj:
            model.blocks[i].attn.proj = MoR_Linear(raw_linear=model.blocks[i].attn.proj,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            
        if lora_mlp:
            if 'mlp_fc1' in lora_layers:
                model.blocks[i].mlp.fc1 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc1,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            if 'mlp_fc2' in lora_layers:
                model.blocks[i].mlp.fc2 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc2,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            


def set_panderm_mor(model, r, alpha, lora_layers, num_loras):
    # lora_layers: q, k, v, proj, mlp
    if 'q' in lora_layers:
        lora_q = True
    else:
        lora_q = False
        
    if 'k' in lora_layers:
        lora_k = True
    else:
        lora_k = False
        
    if 'v' in lora_layers:
        lora_v = True
    else:
        lora_v = False
        
    if 'proj' in lora_layers:
        lora_proj = True
    else:
        lora_proj = False
        
    if 'mlp_fc1' in lora_layers or 'mlp_fc2' in lora_layers:
        lora_mlp = True
    else:
        lora_mlp = False
        
    print('lora_q: {}, lora_k: {}, lora_v: {}, lora_proj: {}, lora_proj: {}, lora_mlp: {}'.format(
        lora_q, lora_k, lora_v, lora_proj, lora_proj, lora_mlp))
    
    for i in range(len(model.blocks)):
        if lora_q or lora_k or lora_v:
            model.blocks[i].attn.qkv = MoR_qkv(raw_qkv_linear=model.blocks[i].attn.qkv, 
                                                   lora_q=lora_q, lora_k=lora_k, lora_v=lora_v,
                                                   r=r, alpha=alpha, num_loras=num_loras
                                                )
        
        if lora_proj:
            model.blocks[i].attn.proj = MoR_Linear(raw_linear=model.blocks[i].attn.proj,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            
        if lora_mlp:
            if 'mlp_fc1' in lora_layers:
                model.blocks[i].mlp.fc1 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc1,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            if 'mlp_fc2' in lora_layers:
                model.blocks[i].mlp.fc2 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc2,
                                                       r=r, alpha=alpha, num_loras=num_loras)
 


def set_urfound_mor(model, r, alpha, lora_layers, num_loras):
    # lora_layers: q, k, v, proj, mlp
    if 'q' in lora_layers:
        lora_q = True
    else:
        lora_q = False
        
    if 'k' in lora_layers:
        lora_k = True
    else:
        lora_k = False
        
    if 'v' in lora_layers:
        lora_v = True
    else:
        lora_v = False
        
    if 'proj' in lora_layers:
        lora_proj = True
    else:
        lora_proj = False
        
    if 'mlp_fc1' in lora_layers or 'mlp_fc2' in lora_layers:
        lora_mlp = True
    else:
        lora_mlp = False
        
    print('lora_q: {}, lora_k: {}, lora_v: {}, lora_proj: {}, lora_proj: {}, lora_mlp: {}'.format(
        lora_q, lora_k, lora_v, lora_proj, lora_proj, lora_mlp))
    
    for i in range(len(model.blocks)):
        if lora_q or lora_k or lora_v:
            model.blocks[i].attn.qkv = MoR_qkv(raw_qkv_linear=model.blocks[i].attn.qkv, 
                                                   lora_q=lora_q, lora_k=lora_k, lora_v=lora_v,
                                                   r=r, alpha=alpha, num_loras=num_loras
                                                )
        
        if lora_proj:
            model.blocks[i].attn.proj = MoR_Linear(raw_linear=model.blocks[i].attn.proj,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            
        if lora_mlp:
            if 'mlp_fc1' in lora_layers:
                model.blocks[i].mlp.fc1 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc1,
                                                       r=r, alpha=alpha, num_loras=num_loras)
            if 'mlp_fc2' in lora_layers:
                model.blocks[i].mlp.fc2 = MoR_Linear(raw_linear=model.blocks[i].mlp.fc2,
                                                       r=r, alpha=alpha, num_loras=num_loras)
 


