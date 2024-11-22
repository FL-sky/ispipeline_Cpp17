
/*
USM： (src - w * gauss）/（1 - w）
w: 0.1～0.9, default: 0.6
*/
#include "modules/modules.h"

#define MOD_NAME "sharpen"

const int KernelSum = 273;
const int kGaussKernel[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1},
};

static int SharpenGauss(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;

    uint8_t *y_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.y);
    uint8_t *y_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.y);

    float ratio = isp_prm->sharpen_prms.ratio;

    FOR_ITER(h, frame->info.height)
    {
        FOR_ITER(w, frame->info.width)
        {
            pixel_idx = h * frame->info.width + w;
            if ((w < 2) || (h < 2) || (w > (frame->info.width - 3)) || (h > (frame->info.height - 3)))
            {
                y_o[pixel_idx] = y_i[pixel_idx];
                continue;
            }

            int y = 0;

            for (int kh = h - 2, gauss_idy = 0; kh <= h + 2; ++kh, ++gauss_idy)
            {
                for (int kw = w - 2, gauss_idx = 0; kw <= w + 2; ++kw, ++gauss_idx)
                {
                    y += (y_i[GET_PIXEL_INDEX(kw, kh, frame->info.width)] * kGaussKernel[gauss_idy][gauss_idx]);
                }
            }
            y = y / KernelSum;

            y = static_cast<int>((y_i[pixel_idx] - ratio * y) / (1 - ratio));

            ClipMinMax(y, 255, 0);
            y_o[pixel_idx] = y;
        }
    }

    SwapMem<void>(frame->data.yuv_u8_i.y, frame->data.yuv_u8_o.y);

    return 0;
}

//(Frame *frame, const IspPrms *isp_prm)
//    BilateralFilter(y_log, bp_log, 9, frame->info.width, frame->info.height, ltm_prms.space_sigma, ltm_prms.range_sigma, max_log_base, min_log_base);
static int SharpenBilateral(Frame *frame, const IspPrms *isp_prm)
{
    // float *y_i, float *bp_log, int kernel_size,
    //                         int width, int height, float space_sigma,
    //                         float range_sigma
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;

    uint8_t *y_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.y);
    uint8_t *y_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.y);

    float g_guass_kernel[kMaxLtmKenerlSize][kMaxLtmKenerlSize];
    float g_range_kernel[kMaxLtmKenerlSize][kMaxLtmKenerlSize];
    float g_final_kernel[kMaxLtmKenerlSize][kMaxLtmKenerlSize];
    int kernel_size = 5; //
    if (kernel_size > kMaxLtmKenerlSize)
    {
        kernel_size = kMaxLtmKenerlSize;
    }
    // std::cout << kernel_size << "\n";
    int center = kernel_size >> 1;
    int space_sigma = 20; //

    float sigma_2 = space_sigma * space_sigma;

    for (int kh = 0; kh < kernel_size; ++kh)
    {
        for (int kw = 0; kw < kernel_size; ++kw)
        {
            float exp_scale = -0.5f * ((kh - center) * (kh - center) + (kw - center) * (kw - center)) / sigma_2;
            g_guass_kernel[kw][kh] = exp(exp_scale);
            // printf("[%f]", g_guass_kernel[kw][kh]);
        }
        // printf("\n");
    }
    int range_sigma = 20; //
    sigma_2 = range_sigma * range_sigma;
    int width = frame->info.width;
    int height = frame->info.height;
    FOR_ITER(ih, frame->info.height)
    {
        FOR_ITER(iw, frame->info.width)
        {
            // 每个像素做filter
            float y_gray = 0;
            float filter_kernel_sum = 0;
            float filter_result = 0;
            int pixel_id = GET_PIXEL_INDEX(iw, ih, width);

            for (int kh = -center; kh <= center; ++kh)
            {
                for (int kw = -center; kw <= center; ++kw)
                {
                    int idx = iw + kw;
                    int idy = ih + kh;

                    if ((idx < 0) && (idy < 0))
                    {
                        y_gray = y_i[0];
                    }
                    else if ((idx > 0) && (idy < 0))
                    {
                        if (idx < width)
                        {
                            y_gray = y_i[idx];
                        }
                        else
                        {
                            y_gray = y_i[width - 1];
                        }
                    }
                    else if ((idx < 0) && (idy > 0))
                    {
                        if (idy < height)
                        {
                            y_gray = y_i[idy * width];
                        }
                        else
                        {
                            y_gray = y_i[(height - 1) * width];
                        }
                    }
                    else if ((idx >= 0) && (idy >= 0))
                    {
                        if ((idx < width) && (idy < height))
                        {
                            y_gray = y_i[idx + idy * width];
                        }
                        else if ((idx >= width) && (idy < height))
                        {
                            y_gray = y_i[(width - 1) + idy * width];
                        }
                        else if ((idx >= width) && (idy >= height))
                        {
                            y_gray = y_i[height * width - 1];
                        }
                        else if ((idx < width) && (idy >= height))
                        {
                            y_gray = y_i[(height - 1) * width + idx];
                        }
                        else
                        {
                            LOG(ERROR) << "error padding";
                            y_gray = 0;
                        }
                    }

                    float exp_scale = (-0.5 * (y_gray - y_i[pixel_id]) * (y_gray - y_i[pixel_id])) / sigma_2;
                    g_range_kernel[kw + center][kh + center] = exp(exp_scale);

                    g_final_kernel[kw + center][kh + center] = g_guass_kernel[kw + center][kh + center] * g_range_kernel[kw + center][kh + center];

                    filter_result += (g_final_kernel[kw + center][kh + center] * y_gray);
                    filter_kernel_sum += g_final_kernel[kw + center][kh + center];
                }
            }
            filter_result = filter_result / filter_kernel_sum;

            // detail = (y_i[pixel_id] - filter_result);
            int weight = 1.5;//3
            y_o[pixel_id] = filter_result + weight * (y_i[pixel_id] - filter_result);
            /////###y_out = (1-weight_ratio)*y_bilateral_filtered + (weight_ratio=0.3) * detail
            // y_o[pixel_id] = filter_result;
            if (y_o[pixel_id] < 0)
                y_o[pixel_id] = 0;
            if (y_o[pixel_id] > 255)
                y_o[pixel_id] = 255;
        }
    }
    SwapMem<void>(frame->data.yuv_u8_i.y, frame->data.yuv_u8_o.y);
    return 0;
}

void RegisterSharpenMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::YUV;
    mod.out_domain = ColorDomains::YUV;

    mod.name = MOD_NAME;
    mod.run_function = SharpenBilateral;

    RegisterIspModule(mod);
}