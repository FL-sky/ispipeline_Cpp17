#include "modules/modules.h"

#define MOD_NAME "cns"

static int CnsMedian(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;

    uint8_t *u_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.u);
    uint8_t *v_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.v);
    uint8_t *u_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.u);
    uint8_t *v_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.v);

    constexpr int boundary_pixel = 1;
    constexpr int filter_size = (2 * boundary_pixel + 1) * (2 * boundary_pixel + 1);
    constexpr int filter_center = filter_size >> 1;
    uint8_t u[filter_size];
    uint8_t v[filter_size];

    FOR_ITER(ih, frame->info.height)
    {
        FOR_ITER(iw, frame->info.width)
        {
            int pixel_idx = ih * frame->info.width + iw;
            if ((iw < boundary_pixel) || (iw >= (frame->info.width - boundary_pixel)) || (ih < boundary_pixel) || (ih >= (frame->info.height - boundary_pixel)))
            {
                u_o[pixel_idx] = u_i[pixel_idx];
                v_o[pixel_idx] = v_i[pixel_idx];
                continue;
            }

            int sub_index = 0;
            for (int idy = -boundary_pixel; idy <= boundary_pixel; ++idy)
            {
                for (int idx = -boundary_pixel; idx <= boundary_pixel; ++idx)
                {
                    int filer_pixel_idx = GET_PIXEL_INDEX((iw + idx), (ih + idy), frame->info.width);
                    u[sub_index] = u_i[filer_pixel_idx];
                    v[sub_index] = v_i[filer_pixel_idx];
                    ++sub_index;
                }
            }
            // meida filter
            std::sort(u, u + filter_size);
            std::sort(v, v + filter_size);

            u_o[pixel_idx] = u[filter_center];
            v_o[pixel_idx] = v[filter_center];
        }
    }

    SwapMem<void>(frame->data.yuv_u8_i.u, frame->data.yuv_u8_o.u);
    SwapMem<void>(frame->data.yuv_u8_i.v, frame->data.yuv_u8_o.v);

    return 0;
}

void cg_padding_symmetric(int &h, int &w, int height, int width)
{
    if (h < 0)
        h = abs(h + 1);
    else if (h >= height)
        h = height - 1 - (h - height);

    if (w < 0)
        w = abs(w + 1);
    else if (w >= width)
        w = width - 1 - (w - width);
}

uint64_t sqr_64(uint64_t x)
{
    return x * x;
}

uint64_t getDistance(uint8_t *src, int height, int width, int filter_size, int x0, int y0, int x1, int y1)
{
    uint64_t ret = 0;
    for (int i = -filter_size; i <= filter_size; i++)
    {
        for (int j = -filter_size; j <= filter_size; j++)
        {
            int a = x0 + i, b = y0 + j;
            int c = x1 + i, d = y1 + j;
            cg_padding_symmetric(a, b, height, width);
            cg_padding_symmetric(c, d, height, width);
            int p0 = GET_PIXEL_INDEX(b, a, width);
            int p1 = GET_PIXEL_INDEX(d, c, width);
            ret += sqr_64(src[p0] - src[p1]);
        }
    }
    return ret;
}

static int YuvNlm(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;

    uint8_t *y_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.y);
    uint8_t *u_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.u);
    uint8_t *v_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.v);
    uint8_t *y_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.y);
    uint8_t *u_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.u);
    uint8_t *v_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.v);

    int height = frame->info.height;
    int width = frame->info.width;
//    float kernel_cov = 5.0; // #kernel_cov可调系数越大越模糊
    float kernel_cov = 3.0; // #kernel_cov可调系数越大越模糊
    // #多大的块去比
//    int filter_size = 3; //  # the radio of the filter
    int filter_size = 2; //  # the radio of the filter
    float filter_size2 = filter_size * filter_size;
    // #search windows
//    int search_size = 10; //  # the ratio of the search size
    int search_size = 7; //  # the ratio of the search size
    // #边缘扩充
    // pad_img = np.pad(img, ((filter_size, filter_size), (filter_size, filter_size)), 'symmetric')
    // result = np.zeros(img.shape)
    // # 归一化的
    // kernel = np.ones((2 * filter_size + 1, 2 * filter_size + 1))
    // kernel = kernel / ((2 * filter_size + 1) ** 2)

    for (int h = 0; h < height; h++)
    {
//        printf("cns h=%d\n",h);
        for (int w = 0; w < width; w++)
        {
            /// x_pixels = pad_img[w1-filter_size:w1+filter_size+1, h1-filter_size:h1+filter_size+1]
            // # x_pixels = np.reshape(x_pixels, (49, 1)).squeeze()
            int w_min = Max(w - search_size, 0);
            int w_max = Min(w + search_size, width - 1);
            int h_min = Max(h - search_size, 0);
            int h_max = Min(h + search_size, height - 1);
            double ysum_similarity = 0, ysum_pixel = 0, yweight_max = 0;
            double usum_similarity = 0, usum_pixel = 0, uweight_max = 0;
            double vsum_similarity = 0, vsum_pixel = 0, vweight_max = 0;
            for (int x = w_min; x <= w_max; x++)
            {
                for (int y = h_min; y <= h_max; y++)
                {
                    if (x == w && y == h)
                        continue;
                    // #y_pixels块
                    // y_pixels = pad_img[x-filter_size:x+filter_size+1, y-filter_size:y+filter_size+1]
                    // #块中所有点的距离都算出来了
                    uint64_t ydis2 = getDistance(y_i, height, width, filter_size, h, w, y, x);
                    uint64_t udis2 = getDistance(u_i, height, width, filter_size, h, w, y, x);
                    uint64_t vdis2 = getDistance(v_i, height, width, filter_size, h, w, y, x);
                    // distance = x_pixels - y_pixels;
                    // distance = np.sum(np.multiply(kernel, np.square(distance)));
                    double ydis = ydis2 / filter_size2;
                    double udis = udis2 / filter_size2;
                    double vdis = vdis2 / filter_size2;
                    // #相似度就是权重;
                    double ysimilarity = exp(-ydis / (kernel_cov * kernel_cov));
                    double usimilarity = exp(-udis / (kernel_cov * kernel_cov));
                    double vsimilarity = exp(-vdis / (kernel_cov * kernel_cov));
                    // #;
                    if (ysimilarity > yweight_max)
                        yweight_max = ysimilarity;
                    if (usimilarity > uweight_max)
                        uweight_max = usimilarity;
                    if (vsimilarity > vweight_max)
                        vweight_max = vsimilarity;
                    ysum_similarity += ysimilarity;
                    usum_similarity += usimilarity;
                    vsum_similarity += vsimilarity;
                    ysum_pixel += ysimilarity * y_i[GET_PIXEL_INDEX(x, y, width)];
                    usum_pixel += usimilarity * u_i[GET_PIXEL_INDEX(x, y, width)];
                    vsum_pixel += vsimilarity * v_i[GET_PIXEL_INDEX(x, y, width)];
                }
            }
            ysum_pixel += y_i[GET_PIXEL_INDEX(w, h, width)];
            usum_pixel += u_i[GET_PIXEL_INDEX(w, h, width)];
            vsum_pixel += v_i[GET_PIXEL_INDEX(w, h, width)];
            ysum_similarity += 1;
            usum_similarity += 1;
            vsum_similarity += 1;
            if (ysum_similarity > 0)
                y_o[GET_PIXEL_INDEX(w, h, width)] = ysum_pixel / ysum_similarity;
            else
                y_o[GET_PIXEL_INDEX(w, h, width)] = y_i[GET_PIXEL_INDEX(w, h, width)];
            if (usum_similarity > 0)
                u_o[GET_PIXEL_INDEX(w, h, width)] = usum_pixel / usum_similarity;
            else
                u_o[GET_PIXEL_INDEX(w, h, width)] = u_i[GET_PIXEL_INDEX(w, h, width)];
            if (vsum_similarity > 0)
                v_o[GET_PIXEL_INDEX(w, h, width)] = vsum_pixel / vsum_similarity;
            else
                v_o[GET_PIXEL_INDEX(w, h, width)] = v_i[GET_PIXEL_INDEX(w, h, width)];
        }
    }
    SwapMem<void>(frame->data.yuv_u8_i.y, frame->data.yuv_u8_o.y);
    SwapMem<void>(frame->data.yuv_u8_i.u, frame->data.yuv_u8_o.u);
    SwapMem<void>(frame->data.yuv_u8_i.v, frame->data.yuv_u8_o.v);
    return 0;
}

void RegisterCnsMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::YUV;
    mod.out_domain = ColorDomains::YUV;

    mod.name = MOD_NAME;
    mod.run_function = CnsMedian;

    RegisterIspModule(mod);
}