# Deep Learning Model Performance Analysis

## OVERALL BEST PERFORMERS (Aggregated Mean)
| Metric | Model + Wavelet | Value | Runtime (s) |
|:---|:---|:---|:---|
| Lowest MSE | SWIN + haar | 0.000677 | 5691 |
| Highest SSIM | SWIN + Baseline (None) | 0.9638 | 10195 |


## BASELINE PERFORMANCE (No Wavelet - 'Baseline (None)')
| model_type   |   eval/mse_mean |   eval/ssim_mean |   Runtime |
|:-------------|----------------:|-----------------:|----------:|
| UNETR        |        0.000677 |         0.961028 |     10308 |
| SWIN         |        0.000709 |         0.963774 |     10195 |
| UNET         |        0.001138 |         0.961369 |      1865 |


## BEST WAVELET CONFIGURATION PER MODEL
| Model   | Best Wavelet    |   Best MSE |   Baseline MSE | Improvement (vs Baseline)   |
|:--------|:----------------|-----------:|---------------:|:----------------------------|
| SWIN    | haar            |   0.000677 |       0.000709 | 4.53%                       |
| UNETR   | Baseline (None) |   0.000677 |       0.000677 | 0.00%                       |
| UNET    | db2             |   0.000803 |       0.001138 | 29.40%                      |


## MODALITY-SPECIFIC BEST PERFORMERS (Lowest MSE)
| Modality | Best Configuration | MSE | SSIM |
|:---|:---|:---|:---|
| FLAIR | UNETR + Baseline (None) | 0.000536 | 0.9654 |
| T1 | UNETR + Baseline (None) | 0.000468 | 0.9673 |
| T1CE | SWIN + haar | 0.000782 | 0.9093 |
| T2 | UNETR + db2 | 0.000801 | 0.9239 |

