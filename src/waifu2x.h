#pragma once

// waifu2x implemented with ncnn library

#include <string>

// ncnn
#include "ncnn/gpu.h"
#include "ncnn/layer.h"
#include "ncnn/net.h"

class Waifu2x
{
public:
    Waifu2x(int gpuid, bool tta_mode = false, int num_threads = 1);
    ~Waifu2x();

#if _WIN32
    int load(const std::wstring& parampath, const std::wstring& modelpath, const bool fp32);
#else
    int load(const std::string& parampath, const std::string& modelpath, const bool fp32);
#endif

    int process(const float* srcR, const float* srcG, const float* srcB,
        float* dstR, float* dstG, float* dstB,
        const int w, const int h, const ptrdiff_t srcStride, const ptrdiff_t dstStride) const;

public:
    // waifu2x parameters
    int noise;
    int scale;
    int tile_w;
    int tile_h;
    int prepadding;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;
    ncnn::Layer* bicubic_2x;
    bool tta_mode;
};
