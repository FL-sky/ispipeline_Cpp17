// ## https: // blog.csdn.net/weixin_42166222/article/details/123377594

#include "modules/modules.h"

#define MOD_NAME "wbgain"

static int WbGain(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;
    int pwl_idx = 0;

    int32_t *raw32_in = reinterpret_cast<int32_t *>(frame->data.raw_s32_i);
    int32_t *raw32_out = reinterpret_cast<int32_t *>(frame->data.raw_s32_o);

    // default d65
    float r_gain = isp_prm->wb_gains.d65_gain[0];
    float gr_gain = isp_prm->wb_gains.d65_gain[1];
    float gb_gain = isp_prm->wb_gains.d65_gain[2];
    float b_gain = isp_prm->wb_gains.d65_gain[3];

    FOR_ITER(h, frame->info.height)
    {
        FOR_ITER(w, frame->info.width)
        {
            pixel_idx = h * frame->info.width + w;

            int cfa_id = static_cast<int>(frame->info.cfa);
            switch (kPixelCfaLut[cfa_id][w % 2][h % 2])
            {
            case PixelCfaTypes::R:
                raw32_out[pixel_idx] = (int32_t)(raw32_in[pixel_idx] * r_gain);
                break;
            case PixelCfaTypes::GR:
                raw32_out[pixel_idx] = (int32_t)(raw32_in[pixel_idx] * gr_gain);
                break;
            case PixelCfaTypes::GB:
                raw32_out[pixel_idx] = (int32_t)(raw32_in[pixel_idx] * gb_gain);
                break;
            case PixelCfaTypes::B:
                raw32_out[pixel_idx] = (int32_t)(raw32_in[pixel_idx] * b_gain);
                break;
            default:
                break;
            }

            ClipMinMax<int32_t>(raw32_out[pixel_idx], (int32_t)isp_prm->info.max_val, 0);
        }
    }

    SwapMem<void>(frame->data.raw_s32_i, frame->data.raw_s32_o);

    return 0;
}

static int WbGainGrayworld(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;
    int pwl_idx = 0;

    int32_t *raw32_in = reinterpret_cast<int32_t *>(frame->data.raw_s32_i);
    int32_t *raw32_out = reinterpret_cast<int32_t *>(frame->data.raw_s32_o);

    long long g4[2][2] = {0, 0, 0, 0};
    FOR_ITER(h, frame->info.height)
    {
        FOR_ITER(w, frame->info.width)
        {
            pixel_idx = h * frame->info.width + w;
            g4[w & 1][h & 1] += raw32_in[pixel_idx];
        }
    }
    auto illum_max = Max(g4[0][0], g4[0][1]);
    illum_max = Max(illum_max, g4[1][0]);
    illum_max = Max(illum_max, g4[1][1]);
    double gain4f[2][2] = {0, 0, 0, 0};
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            gain4f[i][j] = 1.0 * illum_max / g4[i][j];
        }
    }

    FOR_ITER(h, frame->info.height)
    {
        FOR_ITER(w, frame->info.width)
        {
            pixel_idx = h * frame->info.width + w;
            raw32_out[pixel_idx] = (int32_t)(raw32_in[pixel_idx] * gain4f[h & 1][w & 1]);

            ClipMinMax<int32_t>(raw32_out[pixel_idx], (int32_t)isp_prm->info.max_val, 0);
        }
    }

    SwapMem<void>(frame->data.raw_s32_i, frame->data.raw_s32_o);

    return 0;
}

void RegisterWbGaincMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::RAW;
    mod.out_domain = ColorDomains::RAW;

    mod.name = MOD_NAME;
    mod.run_function = WbGainGrayworld;

    RegisterIspModule(mod);
}