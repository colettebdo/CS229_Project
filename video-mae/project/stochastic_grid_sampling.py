# --------------------------------------------------------
# This function was added for the 229 Final Project.
# This function stochastically chooses a infinitely tiling 
# pattern for image reconstruction.
# --------------------------------------------------------

import torch
import random

def grid_masking(x, mask_ratio):
    """
    Perform per-sample grid masking using 25% offset tile.
    x: [N, L, D], sequence
    """
    mask_ratio = 0.75
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    tensor = torch.ones(1, 196)
    tensor[:,::4] = 0
    tensor = tensor.view(1, 14, 14)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(1, 196)

    noise = tensor
    
    # Developed from original sampling / masking configuration so as to be compatible with the models.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def multi_stochastic_grid_masking(x, mask_ratio):
    """
    Perform per-sample grid masking using 25% offset tile.
    x: [N, L, D], sequence
    """
    mask_ratio = 0.75
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    grid_type = random.randint(0, 1)
    
    if grid_type == 0:
        grid_offset = random.randint(0, 3)
        tensor = torch.ones(1, 196)
        tensor[:,grid_offset::4] = 0
    elif grid_type == 1:
        grid_offset = random.randint(0, 3)
        tensor = torch.ones(1, 196)
        tensor[:,grid_offset::4] = 0
        tensor = tensor.view(1, 14, 14)
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(1, 196)
    else:
        rows, cols = 14, 14
        row_indices = torch.arange(rows).unsqueeze(1)  # Shape (14, 1)
        col_indices = torch.arange(cols).unsqueeze(0)  # Shape (1, 14)
        tensor = 1 - (row_indices % 2) * (col_indices % 2)
        tensor = tensor.reshape(1, 196)

    noise = tensor
    
    # Developed from original sampling / masking configuration so as to be compatible with the models.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def vertical_dither_masking(x, mask_ratio):
    """
    Perform per-sample grid masking using 25% offset tile.
    x: [N, L, D], sequence
    """
    mask_ratio = 0.75
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    grid_offset = random.randint(0, 3)
    tensor = torch.ones(1, 196)
    tensor[:,grid_offset::4] = 0
    tensor = tensor.view(1, 14, 14)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(1, 196)

    noise = tensor
    
    # --------------------------------------------------------
    # Developed from original sampling / masking configuration so as to be compatible with the models.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # --------------------------------------------------------

    return x_masked, mask, ids_restore

def horizontal_dither_masking(x, mask_ratio):
    """
    Perform per-sample grid masking using 25% offset tile.
    x: [N, L, D], sequence
    """
    mask_ratio = 0.75
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    grid_offset = random.randint(0, 3)
    tensor = torch.ones(1, 196)
    tensor[:,grid_offset::4] = 0

    noise = tensor
    
    # --------------------------------------------------------
    # Developed from original sampling / masking configuration so as to be compatible with the models.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # --------------------------------------------------------

    return x_masked, mask, ids_restore

def square_grid_masking(x, mask_ratio):
    """
    Perform per-sample grid masking using 25% offset tile.
    x: [N, L, D], sequence
    """
    mask_ratio = 0.75
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
      
    rows, cols = 14, 14
    row_indices = torch.arange(rows).unsqueeze(1)  # Shape (14, 1)
    col_indices = torch.arange(cols).unsqueeze(0)  # Shape (1, 14)
    tensor = 1 - (row_indices % 2) * (col_indices % 2)
    tensor = tensor.reshape(1, 196)

    noise = tensor
    
    # --------------------------------------------------------
    # Developed from original sampling / masking configuration so as to be compatible with the models.
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # --------------------------------------------------------

    return x_masked, mask, ids_restore


