## Description

ncnn implementation of waifu2x converter, based on [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan).

This is [a port of the VapourSynth plugin w2xncnnvk.Waifu2x](https://github.com/HolyWu/vs-waifu2x-ncnn-vulkan).

### Requirements:

- Vulkan device

- AviSynth+ r3682 (can be downloaded from [here](https://gitlab.com/uvz/AviSynthPlus-Builds) until official release is uploaded) (r3689 recommended) or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

`models` must be located in the same folder as `w2xncnnvk`.

```
w2xncnnvk(clip input, int "noise", int "scale", int "tile_w", int "tile_h", int "model", int "gpu_id", int "gpu_thread", bool "tta", bool "fp32", bool "list_gpu")
```

### Parameters:

- input\
    A clip to process.\
    It must be in RGB 32-bit planar format.

- noise\
    Denoise level.\
    Large value means strong denoise effect, -1 - no effect.\
    Must be between -1..3.\
    Default: 0.

- scale\
    Upscale ratio.\
    Must be either 1 or 2.\
    Default: 2.

- tile_w, tile_h\
    Tile width and height, respectively.\
    Use smaller value to reduce GPU memory usage.\
    Must be equal to or greater than 32.\
    Default: input_width, input_height.

- model\
    Model to use.\
    0: upconv_7_anime_style_art_rgb\
    1: upconv_7_photo\
    2: cunet\
    Default: 2.

- gpu_id\
    GPU device to use.\
    By default the default device is selected.

- gpu_thread\
    Thread count for upscaling.\
    Using larger values may increase GPU usage and consume more GPU memory. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.\
    Default: 2.

- tta\
    Enable TTA(Test-Time Augmentation) mode.\
    Default: False.

- fp32\
    Enable FP32 mode.\
    Default: False.

- list_gpu\
    Simply print a list of available GPU devices on the frame and does nothing else.\
    Default: False.

### Building:

- Requires `Boost`, `Vulkan SDK`, `ncnn`.

- Windows\
    Use solution files.
