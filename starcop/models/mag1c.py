#!/usr/bin/env python
#
# This module is a modified version of mag1c (https://github.com/markusfoote/mag1c) that implements different
# models to estimate methane concentrations from hyperspectral images. The following is the licence of the original mag1c
# code
#
#       M ethane detection with
#       A lbedo correction and
# rewei G hted
#     L 1 sparsity
#       C ode
#
# BSD 3-Clause License
#
# Copyright (c) 2019,
#   Scientific Computing and Imaging Institute and
#   Utah Remote Sensing Applications Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Markus Foote (foote@sci.utah.edu)
import rasterio
import torch
import torch.utils.data
from typing import Tuple, Optional, Union, List, Callable
import numpy as np
import spectral
import os
from tqdm import tqdm

NODATA = -9999
SCALING = 1e5
EPSILON = 1e-9


def generate_template_from_bands(centers: Union[np.ndarray, List], fwhm: Union[np.ndarray, List]) -> np.ndarray:
    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers. (K, )
    :param fwhm: full width half maximum for the gaussian kernel of each band. (K, )
    :return template: the unit absorption spectum
    """
    centers = np.asarray(centers)
    fwhm = np.asarray(fwhm)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError('Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).')
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError('Length of band center wavelengths and band fwhm arrays must be equal.')
    lib = spectral.io.envi.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.hdr'),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.lut'))
    rads = np.asarray(lib.asarray()).squeeze() # (7, 31800)
    wave = np.asarray(lib.bands.centers) # (31800,)
    concentrations = np.asarray([0, 500, 1000, 2000, 4000, 8000, 16000]) # (7, )
    # sigma = fwhm / ( 2 * sqrt( 2 * ln(2) ) )  ~=  fwhm / 2.355
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) # (K, )
    # response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
    # Evaluate normal distribution explicitly
    var = sigma ** 2 # (K, )
    denom = (2 * np.pi * var) ** 0.5 # (K, )
    numer = np.exp(-(wave[:, None] - centers[None, :])**2 / (2*var)) # (31800, K)
    response = numer / denom # (31800, K)
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(axis=0), where=response.sum(axis=0) > 0) # (31800, K)
    # implement resampling as matrix multiply
    resampled = rads.dot(response) # (7, K)
    lograd = np.log(resampled, where=resampled > 0) # (7, K)
    lsqmat = np.stack((np.ones_like(concentrations), concentrations)).T # (7, 2)
    slope, _, _, _ = np.linalg.lstsq(lsqmat, lograd, rcond=None) # (2, K)
    spectrum = slope[1, :] * SCALING
    target = np.stack((centers, spectrum)).T  # np.stack((np.arange(spectrum.shape[0]), centers, spectrum)).T
    return target


def get_mask_bad_bands(wave: np.ndarray) -> np.ndarray:
    """Calculates a mask of the wavelengths to keep based on water vapor absorption features.
    Rejects wavelengths: - Below 400 nm
                         - Above 2485 nm
                         - Between 1350-1420 nm (water absorption region)
                         - Between 1800-1945 nm (water absorption region)

    :param wave: Vector of wavelengths to evaluate.
    :return:
    """
    keep_mask = ~(np.logical_or(np.logical_or(wave < 400, wave > 2485),
                                np.logical_or(np.logical_and(wave > 1350,
                                                             wave < 1420),
                                              np.logical_and(wave > 1800,
                                                             wave < 1945))))
    return keep_mask


@torch.no_grad()
def func_by_groups(func:Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]],
                   x:Union[np.core.memmap, np.ndarray],
                   groups: Union[np.array],
                   mask: Optional[Union[np.array]]=None,
                   disable_pbar:bool=False,
                   samples_read:int=50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply matching filter function by groups (normally the samples from the glt file which corresponds to each
    CCD sensor)

    :param func: matching filter function
    :param x: (H, W, C)  radiance tensor
    :param groups: (H, W) tensor of type int
    :param mask: (H, W) boolean mask with valid pixels
    :param samples_read.
    :param disable_pbar do not show pbar

    :return: mf, albedo tensors of dims (H, W)
    """
    groups = np.array(groups)
    albedo_out = torch.tensor(np.zeros(x.shape[:2], dtype=x.dtype) + NODATA)
    mf_out = albedo_out.clone()

    if mask is None:
        assert not isinstance(x,np.core.memmap), "If x is a memmap file provide a mask!"
        mask = np.all(x > NODATA, axis=-1)

    group_uniques = np.sort(np.unique(groups[mask]))
    
    range_process = list(range(0, len(group_uniques), samples_read))

    for idx_group_range_start in tqdm(range_process, total=len(range_process), desc=f"Found {len(group_uniques)} groups. Reading by groups of size {samples_read}",
                                      disable=disable_pbar):
        idx_group_range_ends = min(idx_group_range_start + samples_read, len(group_uniques))
        group_range_start = group_uniques[idx_group_range_start]
        group_range_ends = group_uniques[idx_group_range_ends-1]

        mask_it = (groups >= group_range_start) & (groups <= group_range_ends) & mask
        rows, cols = np.where(mask_it)
        slice_rows_cols = slice(np.min(rows), np.max(rows) + 1), slice(np.min(cols), np.max(cols) + 1)

        # Force reading from the image
        x_for_second_loop = np.array(x[slice_rows_cols])

        for group_idx in group_uniques[idx_group_range_start:idx_group_range_ends]:
            # print(f"Processing group: {group_idx}")

            mask_iter = (groups[slice_rows_cols] == group_idx) & mask[slice_rows_cols]

            if np.sum(mask_iter) <= 10:
                # skip if there are very few pixels to estimate values
                continue

            mf_out_iter, albedo_out_iter = func(torch.tensor(x_for_second_loop[mask_iter, :]).unsqueeze(0))
            mf_out[slice_rows_cols][mask_iter] = mf_out_iter[0,:,0]
            albedo_out[slice_rows_cols][mask_iter] = albedo_out_iter[0,:,0]

    return mf_out, albedo_out

@torch.no_grad()
def acrwl1mf(x: torch.Tensor,
             template: torch.Tensor,
             num_iter: int = 30,
             albedo_override: bool = False,
             zero_override: bool = False,
             sparse_override: bool = False,
             covariance_update_scaling: float = 1.,
             alpha: float = 0.,
             compute_energy:bool= False,
             mask: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[float]]]:
    """Calculate the albedo-corrected reweighted-L1 matched filter on radiance data.

    :param x: Radiance Data to process. [b, p, s] See notes below on format.
    :param template: Target spectrum for detection. [s,]
    :param num_iter: Number of iterations to run.
    :param albedo_override: Do not calculate or apply albedo correction factor.
    :param zero_override: Do not apply non-negativity constraint on matched filter results.
    :param sparse_override: Do not use sparse regularization in iterations when True.
    :param covariance_update_scaling: scalar value controls contribution of previous filter values in removing target
        signal in covariance and mean updates.
    :param alpha: scalar value to perform diagonal scaling of the covariance matrix
    :param mask: An optional mask to mark where data should contribute to covariance and mean. [p,]
    :param compute_energy: Compute energy on each iteration

    :returns mf, albedo

        x must be 3-dimensional:
        batch (columns or groups of columns) x
        pixels (samples) x
        spectrum (bands)

        Notice that the samples dimension must be shared
        by the batch, so only batches/columns with the same number
        of pixels to process may be combined into a batch.
    """
    # Some constants / arrays to preallocate
    N = x.shape[1]  # number of pixels (samples)

    out = rmf(x=x, template=template, alpha=alpha, zero_override=zero_override,
              albedo_override=albedo_override, compute_energy=compute_energy,
              mask=mask, apply_scaling=False)

    mf, R = out[:2]

    if compute_energy:
        energy = [out[2]]

    # R  [b, p, 1]
    # mf [b, p, 1]

    # Initialize target
    template = template.unsqueeze(0).unsqueeze(0)  # [1, 1, s]
    if mask is not None:
        modx = x[:, mask]
    else:
        modx = x.clone()
    target = template * torch.mean(modx, dim=1, keepdim=True)

    # Reweighted L1 Algorithm
    for i in range(num_iter):
        # Re-calculate statistics
        if mask is not None:
            modx = x[:, mask] - covariance_update_scaling * R[:, mask] * mf[:, mask] * target
        else:
            modx = x - covariance_update_scaling * R * mf * target

        mu = torch.mean(modx, dim=1, keepdim=True)  # [b, 1, s]
        target = template * mu  # [b, 1, s]
        modx_minus_mu = modx - mu  # [b, p', s]
        x_minus_mu = x - mu  # [b, p, s]

        C = torch.bmm(torch.transpose(modx_minus_mu, 1, 2), modx_minus_mu) / N  # [b, s, s]
        C = C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)

        cholC = torch.linalg.cholesky(C)
        Cit = torch.cholesky_solve(torch.transpose(target, 1, 2), cholC) # [b, s, 1]

        # Calculate new regularizer weights
        if not sparse_override:  # regularizer pre-defined as zeros.
            # In addition, the regularization is scaled by the albedo factor to decrease the regularization of low-signal
            # regions while increasing the confidence in retrievals over high-signal regions
            regularizer = 1 / (R * (mf + EPSILON))  # [b, p, 1]
            # regularizer = R / ( (mf + epsilon)) # I think this corresponds to sentence above??
        else:
            regularizer = 0

        # Compute matched filter with regularization
        normalizer = torch.bmm(target, Cit) # [b, 1, 1]
        if torch.sum(torch.lt(normalizer, 1)):
            normalizer = normalizer.clamp_(min=1)
        mf = (torch.bmm(x_minus_mu, Cit) - regularizer) / (R * normalizer)
        mf = torch.nn.functional.relu_(mf)
        # Energy
        if compute_energy:
            norm_residual = torch.bmm(x_minus_mu,
                                      torch.cholesky_solve(torch.transpose(x_minus_mu, 1, 2), cholC))
            # energy_reg = torch.sum(R*regularizer*mf)
            energy.append(torch.sum(norm_residual))

    mf *= SCALING
    if compute_energy:
        return mf, R, energy

    return mf, R


@torch.no_grad()
def rmf(x: torch.Tensor,
        template: torch.Tensor,
        alpha: float = 0.,
        zero_override: bool = False,
        compute_energy: bool = False,
        albedo_override: bool = False,
        apply_scaling:bool = True,
        mask: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float]]:
    """

    :param x: Radiance Data to process. [b, p, s] See notes below on format.
    :param template: Target spectrum for detection. [s,]
    :param alpha: scalar value to perform diagonal scaling of the covariance matrix
    :param albedo_override: Do not calculate or apply albedo correction factor.
    :param zero_override: Do not apply non-negativity constraint on matched filter results.
    :param compute_energy: Compute energy (loss function)
    :param apply_scaling: Apply scaling to the mf
    :param mask: An optional mask to mark where data should contribute to covariance and mean. [p,]

    :returns: mf result with shape [b, p, 1] and units ppm x m
    """
    N = x.shape[1]  # number of pixels (samples)
    template = template.unsqueeze(0).unsqueeze(0)  # [1, 1, s]
    
    if mask is not None:
        mask = torch.squeeze(mask, 0)
        assert mask.shape == x.shape[1:2], f"Unexpected shape of mask: {mask.shape} expected {x.shape[1:2]}"
        modx = x[:, mask]  # torch.zeros_like(x, dtype=dtype, device=device, layout=torch.strided)
    else:
        modx = x.clone()

    mu = torch.mean(modx, 1, keepdim=True)  # [b ,1 ,s]
    target = template * mu
    x_minus_mu = x - mu

    modx_minus_mu = modx - mu
    C = torch.bmm(torch.transpose(modx_minus_mu, 1, 2), modx_minus_mu) / N  # [b x s x p] * [b x p x s] = [b x s x s]
    C = C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)  # C = (1-alpha) * S + alpha * diag(S)
    # Cit, _ = torch.gesv(torch.transpose(target, 1, 2), C)  # [b x s x 1] \ [b x s x s] = [b x s x 1]
    cholC = torch.linalg.cholesky(C)
    Cit = torch.cholesky_solve(torch.transpose(target, 1, 2), cholC, upper=False) # [b, s, 1]
    normalizer = torch.bmm(target, Cit)  # [b x 1 x s] * [b x s x 1] = [b x 1 x 1]

    if albedo_override:
        R = 1
    else:
        R = torch.bmm(x, torch.transpose(mu, 1, 2)) / torch.bmm(mu, torch.transpose(mu, 1, 2)) # [b, p, 1]

    mf = torch.bmm(x_minus_mu, Cit) / (R * normalizer) # [b, p, 1]

    if not zero_override:
        mf = torch.nn.functional.relu_(mf)  # max(mf, 0)

    if compute_energy:
        norm_residual = torch.sum(torch.bmm(x_minus_mu,torch.cholesky_solve(torch.transpose(x_minus_mu, 1, 2), cholC))).item()
        # TODO add covariance part to the loss!
        det_covariances = 1/torch.prod(torch.diagonal(cholC,dim1=-2,dim2=-1))
        norm_residual+= N/2*torch.log(det_covariances)

        return mf, R, norm_residual

    if apply_scaling:
        mf = mf * SCALING

    return  mf, R

