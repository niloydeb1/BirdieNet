from fastai.vision.all import *

def generate_batch_attention_maps(attn_wgts, targ_sz=None, mode=None):
    
    device = attn_wgts[0][0].device
    batch_size = attn_wgts[0][0].shape[0]

    stage_attentions = []
    final_tokens = attn_wgts[-1][0].shape[-1]
    final_grid_size = int(np.sqrt(final_tokens - 1))  # assumes CLS token in final stage

    for stage_idx, stage_attns in enumerate(attn_wgts):
        # stack block-level attention weights for this stage
        att_mat = torch.stack(stage_attns, dim=1)  # [B, L, H, T, T]
        att_mat = att_mat.mean(dim=2)              # [B, L, T, T] (avg heads)

        # Add identity and normalize
        I = torch.eye(att_mat.size(-1), device=device).unsqueeze(0).unsqueeze(0)
        aug_att_mat = att_mat + I
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

        # Recursive multiplication over layers
        joint_attn = aug_att_mat[:, 0]
        for i in range(1, aug_att_mat.size(1)):
            joint_attn = torch.bmm(aug_att_mat[:, i], joint_attn)

        # CLS token handling
        if stage_idx == len(attn_wgts) - 1:  # final stage
            joint_attn = joint_attn[:, 0, 1:]
        else:
            joint_attn = joint_attn.mean(dim=1)  # no CLS: average over all tokens

        grid_size = int(np.sqrt(joint_attn.shape[-1]))
        joint_attn = joint_attn.view(batch_size, grid_size, grid_size)
        joint_attn /= joint_attn.amax(dim=(-2, -1), keepdim=True)

        # Resize to final stage grid size
        if grid_size != final_grid_size:
            joint_attn = F.interpolate(joint_attn.unsqueeze(1), size=(final_grid_size, final_grid_size),
                                       mode='bilinear', align_corners=True).squeeze(1)

        stage_attentions.append(joint_attn)

    # Equal weights (optional: make learnable)
    weights = torch.softmax(torch.ones(len(stage_attentions), device=device), dim=0)
    combined_attention = sum(w * a for w, a in zip(weights, stage_attentions))

    # Optional final resize
    if targ_sz and mode in ['bilinear', 'nearest']:
        combined_attention = F.interpolate(combined_attention.unsqueeze(1),
                                           size=(targ_sz, targ_sz),
                                           mode=mode,
                                           align_corners=(mode == 'bilinear')).squeeze(1)

    return combined_attention
