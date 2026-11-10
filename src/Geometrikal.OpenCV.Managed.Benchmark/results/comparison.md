# Managed port vs cv2 benchmark

- .NET: .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64, CPU: 11th Gen Intel Core i9-11900 2.50GHz (16 logical cores)
- cv2 1T SSE no-IPP: cv2 4.5.5, threads=1, ipp=False, OPENCV_CPU_DISABLE='AVX,AVX2,AVX512F,FMA3', Python 3.9.5
- cv2 1T AVX2+IPP: cv2 4.5.5, threads=1, ipp=True, OPENCV_CPU_DISABLE='', Python 3.9.5
- cv2 default: cv2 4.5.5, threads=16, ipp=True, OPENCV_CPU_DISABLE='', Python 3.9.5
- Times are medians per call. Ratio = managed / cv2; above 1.00x means the managed port is slower.
- Managed is single-threaded. Managed MT uses all cores (`--threads all`) and is compared with cv2 default.
- Drawing cases draw 64 shapes per call; cv2 times include the Python loop overhead.

## Contour

| Case | Size | Managed | Managed alloc | cv2 1T SSE no-IPP | cv2 1T AVX2+IPP | cv2 default | Ratio vs cv2 1T SSE no-IPP | Ratio vs cv2 1T AVX2+IPP | Ratio vs cv2 default | Managed MT | Ratio MT vs cv2 default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FindContoursExternalSimple | 256 | 155.3 us | 80 KB | 71.3 us | 72.3 us | 72.7 us | 2.18x | 2.15x | 2.14x | - | - |
| FindContoursListTc89L1 | 256 | 293.2 us | 380 KB | 172.3 us | 166.0 us | 171.6 us | 1.70x | 1.77x | 1.71x | - | - |
| FindContoursTreeNone | 256 | 187.0 us | 237 KB | 148.2 us | 138.3 us | 132.3 us | 1.26x | 1.35x | 1.41x | - | - |
| FindContoursExternalSimple | 1024 | 2.25 ms | 1192 KB | 1.24 ms | 1.25 ms | 1.53 ms | 1.81x | 1.81x | 1.47x | - | - |
| FindContoursListTc89L1 | 1024 | 4.16 ms | 4815 KB | 2.59 ms | 2.72 ms | 3.22 ms | 1.61x | 1.53x | 1.29x | - | - |
| FindContoursTreeNone | 1024 | 3.10 ms | 3509 KB | 2.95 ms | 2.72 ms | 3.11 ms | 1.05x | 1.14x | 1.00x | - | - |

## Drawing

| Case | Size | Managed | Managed alloc | cv2 1T SSE no-IPP | cv2 1T AVX2+IPP | cv2 default | Ratio vs cv2 1T SSE no-IPP | Ratio vs cv2 1T AVX2+IPP | Ratio vs cv2 default | Managed MT | Ratio MT vs cv2 default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CircleFilled | 256 | 64.8 us | 2 KB | 99.3 us | 96.6 us | 101.6 us | 0.65x | 0.67x | 0.64x | - | - |
| CircleOutline | 256 | 1.29 ms | 342 KB | 1.28 ms | 1.16 ms | 1.25 ms | 1.01x | 1.11x | 1.03x | - | - |
| DrawContours | 256 | 39.1 us | 2 KB | 52.3 us | 51.7 us | 52.0 us | 0.75x | 0.76x | 0.75x | - | - |
| EllipseOutline | 256 | 1.05 ms | 342 KB | 1.31 ms | 1.24 ms | 1.23 ms | 0.80x | 0.85x | 0.86x | - | - |
| FillPoly | 256 | 269.1 us | 11 KB | 190.1 us | 185.8 us | 195.4 us | 1.42x | 1.45x | 1.38x | - | - |
| Line1px | 256 | 45.7 us | 2 KB | 62.7 us | 74.0 us | 66.4 us | 0.73x | 0.62x | 0.69x | - | - |
| Line5px | 256 | 286.6 us | 2 KB | 229.3 us | 239.9 us | 226.2 us | 1.25x | 1.19x | 1.27x | - | - |
| Polylines | 256 | 74.4 us | 2 KB | 31.9 us | 32.4 us | 31.5 us | 2.33x | 2.30x | 2.36x | - | - |
| RectangleFilled | 256 | 425.9 us | 2 KB | 309.3 us | 332.2 us | 310.6 us | 1.38x | 1.28x | 1.37x | - | - |
| CircleFilled | 1024 | 360.0 us | 2 KB | 377.3 us | 370.1 us | 359.7 us | 0.95x | 0.97x | 1.00x | - | - |
| CircleOutline | 1024 | 1.33 ms | 342 KB | 1.47 ms | 1.58 ms | 1.47 ms | 0.90x | 0.84x | 0.90x | - | - |
| DrawContours | 1024 | 573.4 us | 10 KB | 788.2 us | 740.3 us | 713.5 us | 0.73x | 0.77x | 0.80x | - | - |
| EllipseOutline | 1024 | 1.45 ms | 342 KB | 1.63 ms | 1.56 ms | 1.65 ms | 0.89x | 0.93x | 0.88x | - | - |
| FillPoly | 1024 | 1.13 ms | 11 KB | 820.2 us | 818.2 us | 759.2 us | 1.37x | 1.38x | 1.48x | - | - |
| Line1px | 1024 | 136.9 us | 2 KB | 108.8 us | 107.8 us | 97.9 us | 1.26x | 1.27x | 1.40x | - | - |
| Line5px | 1024 | 610.4 us | 2 KB | 555.9 us | 560.2 us | 540.2 us | 1.10x | 1.09x | 1.13x | - | - |
| Polylines | 1024 | 325.4 us | 2 KB | 130.0 us | 120.7 us | 121.6 us | 2.50x | 2.69x | 2.68x | - | - |
| RectangleFilled | 1024 | 1.27 ms | 2 KB | 1.14 ms | 1.22 ms | 1.19 ms | 1.11x | 1.03x | 1.06x | - | - |

## Filtering

| Case | Size | Managed | Managed alloc | cv2 1T SSE no-IPP | cv2 1T AVX2+IPP | cv2 default | Ratio vs cv2 1T SSE no-IPP | Ratio vs cv2 1T AVX2+IPP | Ratio vs cv2 default | Managed MT | Ratio MT vs cv2 default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Blur15x15_U8 | 256 | 55.1 us | 64 KB | 53.2 us | 49.2 us | 51.4 us | 1.04x | 1.12x | 1.07x | 48.7 us | 0.95x |
| Blur5x5_U8 | 256 | 27.2 us | 64 KB | 72.4 us | 68.3 us | 76.5 us | 0.38x | 0.40x | 0.36x | 20.6 us | 0.27x |
| CopyMakeBorder16_U8C3 | 256 | 94.0 us | 243 KB | 18.1 us | 18.0 us | 18.5 us | 5.20x | 5.22x | 5.07x | 100.8 us | 5.43x |
| Filter2D11x11_U8 | 256 | 341.8 us | 67 KB | 818.7 us | 550.6 us | 604.6 us | 0.42x | 0.62x | 0.57x | 144.5 us | 0.24x |
| Filter2D3x3_U8 | 256 | 44.0 us | 65 KB | 54.2 us | 37.5 us | 42.3 us | 0.81x | 1.17x | 1.04x | 36.1 us | 0.85x |
| GaussianBlur15x15_F32 | 256 | 140.9 us | 259 KB | 137.2 us | 99.4 us | 113.8 us | 1.03x | 1.42x | 1.24x | 108.6 us | 0.95x |
| GaussianBlur15x15_U8 | 256 | 137.4 us | 65 KB | 236.1 us | 189.0 us | 76.1 us | 0.58x | 0.73x | 1.80x | 75.7 us | 0.99x |
| GaussianBlur5x5_F32 | 256 | 64.2 us | 258 KB | 60.0 us | 41.0 us | 48.3 us | 1.07x | 1.57x | 1.33x | 61.0 us | 1.26x |
| GaussianBlur5x5_U8 | 256 | 52.7 us | 65 KB | 56.1 us | 40.3 us | 21.2 us | 0.94x | 1.31x | 2.49x | 28.4 us | 1.34x |
| GaussianBlur5x5_U8C3 | 256 | 140.8 us | 193 KB | 115.9 us | 90.5 us | 34.9 us | 1.21x | 1.56x | 4.04x | 77.3 us | 2.22x |
| Laplacian3_U8S16 | 256 | 45.1 us | 129 KB | 50.6 us | 18.5 us | 18.2 us | 0.89x | 2.44x | 2.48x | 39.2 us | 2.16x |
| MedianBlur15_U8 | 256 | 3.71 ms | 64 KB | 2.25 ms | 2.28 ms | 2.70 ms | 1.65x | 1.63x | 1.38x | 422.8 us | 0.16x |
| MedianBlur3_U8 | 256 | 15.8 us | 64 KB | 72.4 us | 66.3 us | 71.6 us | 0.22x | 0.24x | 0.22x | 11.5 us | 0.16x |
| MedianBlur5_U8 | 256 | 203.9 us | 64 KB | 507.2 us | 503.6 us | 523.5 us | 0.40x | 0.40x | 0.39x | 115.9 us | 0.22x |
| MedianBlur7_U8 | 256 | 3.61 ms | 64 KB | 1.82 ms | 1.86 ms | 2.02 ms | 1.98x | 1.94x | 1.79x | 369.4 us | 0.18x |
| ScharrDx_U8S16 | 256 | 44.6 us | 129 KB | 41.4 us | 32.0 us | 31.4 us | 1.08x | 1.40x | 1.42x | 43.4 us | 1.39x |
| SepFilter2D5_U8F32 | 256 | 67.6 us | 258 KB | 75.7 us | 51.0 us | 47.4 us | 0.89x | 1.33x | 1.43x | 64.8 us | 1.37x |
| Sobel3Dx_U8S16 | 256 | 43.7 us | 129 KB | 35.1 us | 29.3 us | 29.0 us | 1.24x | 1.49x | 1.51x | 41.1 us | 1.42x |
| Sobel3Dxy_F32 | 256 | 51.2 us | 258 KB | 37.7 us | 30.8 us | 28.1 us | 1.36x | 1.66x | 1.82x | 54.6 us | 1.95x |
| Sobel7Dx_U8S16 | 256 | 77.8 us | 129 KB | 152.0 us | 146.5 us | 161.5 us | 0.51x | 0.53x | 0.48x | 59.3 us | 0.37x |
| Blur15x15_U8 | 1024 | 854.9 us | 1024 KB | 827.9 us | 754.5 us | 768.5 us | 1.03x | 1.13x | 1.11x | 555.9 us | 0.72x |
| Blur5x5_U8 | 1024 | 397.6 us | 1025 KB | 1.28 ms | 1.25 ms | 1.26 ms | 0.31x | 0.32x | 0.32x | 373.3 us | 0.30x |
| CopyMakeBorder16_U8C3 | 1024 | 348.7 us | 3269 KB | 762.8 us | 923.1 us | 731.2 us | 0.46x | 0.38x | 0.48x | 358.2 us | 0.49x |
| Filter2D11x11_U8 | 1024 | 5.51 ms | 1027 KB | 12.63 ms | 8.57 ms | 9.37 ms | 0.44x | 0.64x | 0.59x | 1.58 ms | 0.17x |
| Filter2D3x3_U8 | 1024 | 782.6 us | 1025 KB | 926.7 us | 692.8 us | 692.0 us | 0.84x | 1.13x | 1.13x | 551.5 us | 0.80x |
| GaussianBlur15x15_F32 | 1024 | 4.15 ms | 4101 KB | 3.01 ms | 2.12 ms | 2.23 ms | 1.38x | 1.96x | 1.87x | 1.43 ms | 0.64x |
| GaussianBlur15x15_U8 | 1024 | 2.21 ms | 1025 KB | 2.39 ms | 1.74 ms | 612.0 us | 0.92x | 1.27x | 3.60x | 663.8 us | 1.08x |
| GaussianBlur5x5_F32 | 1024 | 1.16 ms | 4101 KB | 1.40 ms | 1.59 ms | 1.28 ms | 0.83x | 0.73x | 0.90x | 1.10 ms | 0.86x |
| GaussianBlur5x5_U8 | 1024 | 815.0 us | 1025 KB | 797.5 us | 699.9 us | 370.3 us | 1.02x | 1.16x | 2.20x | 427.9 us | 1.16x |
| GaussianBlur5x5_U8C3 | 1024 | 2.50 ms | 3073 KB | 2.26 ms | 1.95 ms | 829.0 us | 1.11x | 1.28x | 3.02x | 1.01 ms | 1.22x |
| Laplacian3_U8S16 | 1024 | 790.8 us | 2049 KB | 1.02 ms | 651.0 us | 594.6 us | 0.77x | 1.21x | 1.33x | 711.1 us | 1.20x |
| MedianBlur15_U8 | 1024 | 26.13 ms | 1024 KB | 29.04 ms | 28.52 ms | 29.73 ms | 0.90x | 0.92x | 0.88x | 4.63 ms | 0.16x |
| MedianBlur3_U8 | 1024 | 235.8 us | 1025 KB | 716.3 us | 604.1 us | 644.8 us | 0.33x | 0.39x | 0.37x | 333.8 us | 0.52x |
| MedianBlur5_U8 | 1024 | 2.11 ms | 1024 KB | 3.63 ms | 3.32 ms | 3.06 ms | 0.58x | 0.63x | 0.69x | 602.8 us | 0.20x |
| MedianBlur7_U8 | 1024 | 26.06 ms | 1024 KB | 30.58 ms | 30.53 ms | 29.99 ms | 0.85x | 0.85x | 0.87x | 5.20 ms | 0.17x |
| ScharrDx_U8S16 | 1024 | 993.2 us | 2049 KB | 770.8 us | 769.0 us | 667.2 us | 1.29x | 1.29x | 1.49x | 688.9 us | 1.03x |
| SepFilter2D5_U8F32 | 1024 | 1.18 ms | 4101 KB | 1.63 ms | 1.40 ms | 1.18 ms | 0.72x | 0.84x | 1.00x | 1.39 ms | 1.17x |
| Sobel3Dx_U8S16 | 1024 | 798.2 us | 2050 KB | 725.6 us | 712.7 us | 601.6 us | 1.10x | 1.12x | 1.33x | 673.6 us | 1.12x |
| Sobel3Dxy_F32 | 1024 | 960.6 us | 4101 KB | 1.06 ms | 1.28 ms | 1.08 ms | 0.91x | 0.75x | 0.89x | 1.04 ms | 0.96x |
| Sobel7Dx_U8S16 | 1024 | 1.33 ms | 2049 KB | 3.08 ms | 2.91 ms | 3.04 ms | 0.43x | 0.46x | 0.44x | 833.1 us | 0.27x |

## Morphology

| Case | Size | Managed | Managed alloc | cv2 1T SSE no-IPP | cv2 1T AVX2+IPP | cv2 default | Ratio vs cv2 1T SSE no-IPP | Ratio vs cv2 1T AVX2+IPP | Ratio vs cv2 default | Managed MT | Ratio MT vs cv2 default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DilateEllipse15_U8 | 256 | 183.6 us | 65 KB | 161.4 us | 115.3 us | 117.1 us | 1.14x | 1.59x | 1.57x | 84.6 us | 0.72x |
| DilateEllipse5_U8 | 256 | 22.9 us | 65 KB | 31.0 us | 23.8 us | 24.3 us | 0.74x | 0.96x | 0.94x | 25.9 us | 1.07x |
| DilateRect3Iter3_U8 | 256 | 25.2 us | 65 KB | 23.8 us | 21.7 us | 22.1 us | 1.06x | 1.16x | 1.14x | 26.3 us | 1.19x |
| ErodeRect15_U8 | 256 | 46.9 us | 65 KB | 30.6 us | 25.1 us | 25.5 us | 1.53x | 1.87x | 1.83x | 56.0 us | 2.19x |
| ErodeRect3_U8 | 256 | 12.8 us | 65 KB | 17.9 us | 17.5 us | 16.1 us | 0.72x | 0.73x | 0.80x | 12.6 us | 0.78x |
| ErodeRect3_U8C3 | 256 | 79.8 us | 193 KB | 29.9 us | 24.7 us | 24.8 us | 2.67x | 3.22x | 3.21x | 83.3 us | 3.35x |
| GradientRect3_U8 | 256 | 30.0 us | 193 KB | 35.3 us | 35.4 us | 35.0 us | 0.85x | 0.85x | 0.86x | 30.4 us | 0.87x |
| OpenEllipse5_U8 | 256 | 49.2 us | 129 KB | 59.2 us | 46.6 us | 44.4 us | 0.83x | 1.06x | 1.11x | 53.9 us | 1.22x |
| DilateEllipse15_U8 | 1024 | 2.89 ms | 1027 KB | 2.35 ms | 1.54 ms | 1.60 ms | 1.23x | 1.87x | 1.81x | 822.7 us | 0.52x |
| DilateEllipse5_U8 | 1024 | 332.4 us | 1027 KB | 498.1 us | 409.2 us | 443.0 us | 0.67x | 0.81x | 0.75x | 232.0 us | 0.52x |
| DilateRect3Iter3_U8 | 1024 | 310.5 us | 1026 KB | 410.9 us | 379.9 us | 391.9 us | 0.76x | 0.82x | 0.79x | 284.6 us | 0.73x |
| ErodeRect15_U8 | 1024 | 667.3 us | 1025 KB | 493.9 us | 438.1 us | 405.7 us | 1.35x | 1.52x | 1.64x | 466.2 us | 1.15x |
| ErodeRect3_U8 | 1024 | 259.3 us | 1026 KB | 392.4 us | 356.8 us | 351.6 us | 0.66x | 0.73x | 0.74x | 255.3 us | 0.73x |
| ErodeRect3_U8C3 | 1024 | 824.6 us | 3077 KB | 867.7 us | 963.6 us | 803.3 us | 0.95x | 0.86x | 1.03x | 632.9 us | 0.79x |
| GradientRect3_U8 | 1024 | 732.1 us | 3077 KB | 776.0 us | 810.2 us | 709.6 us | 0.94x | 0.90x | 1.03x | 681.6 us | 0.96x |
| OpenEllipse5_U8 | 1024 | 697.4 us | 2053 KB | 790.1 us | 606.9 us | 612.7 us | 0.88x | 1.15x | 1.14x | 476.0 us | 0.78x |
