"""
Models for middleslice reconstruction
"""

from .wavelet_diffusion import WaveletDiffusion

# Keep backward compatibility
WaveletDiffusionHaar = lambda **kwargs: WaveletDiffusion(wavelet_name='haar', **kwargs)
WaveletDiffusionDb2 = lambda **kwargs: WaveletDiffusion(wavelet_name='db2', **kwargs)

__all__ = ['WaveletDiffusion', 'WaveletDiffusionHaar', 'WaveletDiffusionDb2']