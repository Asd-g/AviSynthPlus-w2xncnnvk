/*
    MIT License

    Copyright (c) 2022 HolyWu
    Copyright (c) 2022 Asd-g

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <algorithm>
#include <atomic>
#include <fstream>
#include <memory>
#include <semaphore>
#include <string>
#include <vector>

#include "avisynth_c.h"
#include "boost/dll/runtime_symbol_info.hpp"
#include "waifu2x.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct w2xncnnvk
{
    AVS_FilterInfo* fi;
    std::unique_ptr<Waifu2x> waifu2x;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
    std::unique_ptr<char[]> msg;
};

static void filter(const AVS_VideoFrame* src, AVS_VideoFrame* dst, const w2xncnnvk* const __restrict d) noexcept
{
    const auto width{ avs_get_row_size_p(src, AVS_PLANAR_B) / avs_component_size(&d->fi->vi) };
    const auto height{ avs_get_height_p(src, AVS_PLANAR_B) };
    const auto srcStride{ avs_get_pitch_p(src, AVS_PLANAR_B) / avs_component_size(&d->fi->vi) };
    const auto dstStride{ avs_get_pitch_p(dst, AVS_PLANAR_B) / avs_component_size(&d->fi->vi) };
    const auto srcR{ reinterpret_cast<const float*>(avs_get_read_ptr_p(src, AVS_PLANAR_R)) };
    const auto srcG{ reinterpret_cast<const float*>(avs_get_read_ptr_p(src, AVS_PLANAR_G)) };
    const auto srcB{ reinterpret_cast<const float*>(avs_get_read_ptr_p(src, AVS_PLANAR_B)) };
    auto dstR{ reinterpret_cast<float*>(avs_get_write_ptr_p(dst, AVS_PLANAR_R)) };
    auto dstG{ reinterpret_cast<float*>(avs_get_write_ptr_p(dst, AVS_PLANAR_G)) };
    auto dstB{ reinterpret_cast<float*>(avs_get_write_ptr_p(dst, AVS_PLANAR_B)) };

    d->semaphore->acquire();
    d->waifu2x->process(srcR, srcG, srcB, dstR, dstG, dstB, width, height, srcStride, dstStride);
    d->semaphore->release();
}

static AVS_VideoFrame* AVSC_CC w2xncnnvk_get_frame(AVS_FilterInfo* fi, int n)
{
    w2xncnnvk* d{ static_cast<w2xncnnvk*>(fi->user_data) };
    auto src{ avs_get_frame(fi->child, n) };
    auto dst{ avs_new_video_frame_p(fi->env, &fi->vi, src) };

    filter(src, dst, d);

    avs_release_video_frame(src);

    return dst;
}

static void AVSC_CC free_w2xncnnvk(AVS_FilterInfo* fi)
{
    auto d{ static_cast<w2xncnnvk*>(fi->user_data) };
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static int AVSC_CC w2xncnnvk_set_cache_hints(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}

static AVS_Value AVSC_CC Create_w2xncnnvk(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum { Clip, Noise, Scale, Tile_w, Tile_h, Model, Gpu_id, Gpu_thread, Tta, Fp32, List_gpu };

    auto d{ new w2xncnnvk() };

    AVS_Clip* clip{ avs_new_c_filter(env, &d->fi, avs_array_elt(args, Clip), 1) };
    AVS_Value v{ avs_void };

    try {
        if (!avs_is_planar(&d->fi->vi) ||
            !avs_is_rgb(&d->fi->vi) ||
            avs_component_size(&d->fi->vi) < 4)
            throw "only RGB 32-bit planar format supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;

        const auto noise{ avs_defined(avs_array_elt(args, Noise)) ? avs_as_int(avs_array_elt(args, Noise)) : 0 };
        const auto scale{ avs_defined(avs_array_elt(args, Scale)) ? avs_as_int(avs_array_elt(args, Scale)) : 2 };
        const auto tile_w{ avs_defined(avs_array_elt(args, Tile_w)) ? avs_as_int(avs_array_elt(args, Tile_w)) : (std::max)(d->fi->vi.width, 32) };
        const auto tile_h{ avs_defined(avs_array_elt(args, Tile_h)) ? avs_as_int(avs_array_elt(args, Tile_h)) : (std::max)(d->fi->vi.height, 32) };
        const auto model{ avs_defined(avs_array_elt(args, Model)) ? avs_as_int(avs_array_elt(args, Model)) : 2 };
        const auto gpuId{ avs_defined(avs_array_elt(args, Gpu_id)) ? avs_as_int(avs_array_elt(args, Gpu_id)) : ncnn::get_default_gpu_index() };
        const auto gpuThread{ avs_defined(avs_array_elt(args, Gpu_thread)) ? avs_as_int(avs_array_elt(args, Gpu_thread)) : 2 };
        const auto tta{ avs_defined(avs_array_elt(args, Tta)) ? avs_as_bool(avs_array_elt(args, Tta)) : 0 };
        const auto fp32{ avs_defined(avs_array_elt(args, Fp32)) ? avs_as_bool(avs_array_elt(args, Fp32)) : 0 };

        if (noise < -1 || noise > 3)
            throw "noise must be between -1 and 3 (inclusive)";
        if (scale < 1 || scale > 2)
            throw "scale must be 1 or 2";
        if (tile_w < 32)
            throw "tile_w must be at least 32";
        if (tile_h < 32)
            throw "tile_h must be at least 32";
        if (model < 0 || model > 2)
            throw "model must be between 0 and 2 (inclusive)";
        if (model != 2 && scale == 1)
            throw "only cunet model supports scale=1";
        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";
        if (auto queue_count{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; gpuThread < 1 || static_cast<uint32_t>(gpuThread) > queue_count)
            throw ("gpu_thread must be between 1 and " + std::to_string(queue_count) + " (inclusive)").c_str();

        if (avs_defined(avs_array_elt(args, List_gpu)) ? avs_as_bool(avs_array_elt(args, List_gpu)) : 0)
        {
            std::string text;

            for (auto i{ 0 }; i < ncnn::get_gpu_count(); ++i)
                text += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

            d->msg = std::make_unique<char[]>(text.size() + 1);
            strcpy(d->msg.get(), text.c_str());

            AVS_Value cl{ avs_new_value_clip(clip) };
            AVS_Value args_[2]{ cl , avs_new_value_string(d->msg.get()) };
            AVS_Value inv{ avs_invoke(d->fi->env, "Text", avs_new_value_array(args_, 2), 0) };
            AVS_Clip* clip1{ avs_new_c_filter(env, &d->fi, inv, 1) };

            v = avs_new_value_clip(clip1);

            avs_release_clip(clip1);
            avs_release_value(inv);
            avs_release_value(cl);
            avs_release_clip(clip);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();

            return v;
        }

        if (noise == -1 && scale == 1)
        {
            v = avs_new_value_clip(clip);

            avs_release_clip(clip);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();

            return v;
        }

        d->fi->vi.width *= scale;
        d->fi->vi.height *= scale;

        auto modelDir{ boost::dll::this_line_location().parent_path().generic_string() + "/models" };
        int prepadding{};

        switch (model)
        {
            case 0:
            {
                modelDir += "/models-upconv_7_anime_style_art_rgb";
                prepadding = 7;
                break;
            }
            case 1:
            {
                modelDir += "/models-upconv_7_photo";
                prepadding = 7;
                break;
            }
            case 2:
            {
                modelDir += "/models-cunet";
                prepadding = (noise == -1 || scale == 2) ? 18 : 28;
                break;
            }
        }

        std::string paramPath, modelPath;

        if (noise == -1)
        {
            paramPath = modelDir + "/scale2.0x_model.param";
            modelPath = modelDir + "/scale2.0x_model.bin";
        }
        else if (scale == 1)
        {
            paramPath = modelDir + "/noise" + std::to_string(noise) + "_model.param";
            modelPath = modelDir + "/noise" + std::to_string(noise) + "_model.bin";
        }
        else
        {
            paramPath = modelDir + "/noise" + std::to_string(noise) + "_scale2.0x_model.param";
            modelPath = modelDir + "/noise" + std::to_string(noise) + "_scale2.0x_model.bin";
        }

        std::ifstream ifs{ paramPath };
        if (!ifs.is_open())
            throw "failed to load model";
        ifs.close();

        d->waifu2x = std::make_unique<Waifu2x>(gpuId, tta, 1);

#ifdef _WIN32
        const auto paramBufferSize{ MultiByteToWideChar(CP_UTF8, 0, paramPath.c_str(), -1, nullptr, 0) };
        const auto modelBufferSize{ MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0) };
        std::vector<wchar_t> wparamPath(paramBufferSize);
        std::vector<wchar_t> wmodelPath(modelBufferSize);
        MultiByteToWideChar(CP_UTF8, 0, paramPath.c_str(), -1, wparamPath.data(), paramBufferSize);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wmodelPath.data(), modelBufferSize);
        d->waifu2x->load(wparamPath.data(), wmodelPath.data(), fp32);
#else
        d->waifu2x->load(paramPath, modelPath, fp32);
#endif

        d->waifu2x->noise = noise;
        d->waifu2x->scale = scale;
        d->waifu2x->tile_w = tile_w;
        d->waifu2x->tile_h = tile_h;
        d->waifu2x->prepadding = prepadding;

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
    }
    catch (const char* error)
    {
        const std::string err{ "waifu2x_nvk: "s + error };
        d->msg = std::make_unique<char[]>(err.size() + 1);
        strcpy(d->msg.get(), err.c_str());
        v = avs_new_value_error(d->msg.get());

        if (--numGPUInstances == 0)
            ncnn::destroy_gpu_instance();
    }

    if (!avs_defined(v))
    {
        v = avs_new_value_clip(clip);

        d->fi->user_data = reinterpret_cast<void*>(d);
        d->fi->get_frame = w2xncnnvk_get_frame;
        d->fi->set_cache_hints = w2xncnnvk_set_cache_hints;
        d->fi->free_filter = free_w2xncnnvk;
    }

    avs_release_clip(clip);

    return v;
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "w2xncnnvk", "c[noise]i[scale]i[tile_w]i[tile_h]i[model]i[gpu_id]i[gpu_thread]i[tta]b[fp32]b[list_gpu]b", Create_w2xncnnvk, 0);
    return "waifu2x ncnn Vulkan";
}
