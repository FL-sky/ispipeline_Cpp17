#include "modules/modules.h"

#define MOD_NAME "dehaze"


//def DarkChannel(im,sz):
//    b,g,r = cv2.split(im)
//    dc = cv2.min(cv2.min(r,g),b);
//    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
//    dark = cv2.erode(dc,kernel)
//    return dark
#include <vector>
using namespace  std;

// 手动实现腐蚀操作
vector<vector<double>> Erode(const vector<vector<double>> &dc, int sz)
{
    int rows = dc.size();
    int cols = dc[0].size();
    vector<vector<double>> dark(rows, vector<double>(cols, 255));

    int half_sz = sz / 2;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double min_val = 255;
            for (int k = -half_sz; k <= half_sz; ++k)
            {
                for (int l = -half_sz; l <= half_sz; ++l)
                {
                    int ni = i + k;
                    int nj = j + l;
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols)
                    {
                        min_val = min(min_val, dc[ni][nj]);
                    }
                }
            }
            dark[i][j] = min_val;
        }
    }

    return dark;
}

vector<vector<double>> DarkChannel(vector<vector<vector<double>>> imbgr, int sz)
{
    int h = imbgr.size();
    int w = imbgr[0].size();
    vector<vector<double>> dc(h);
    FOR_ITER(ih, h)
    {
        dc[ih].resize(w);
        FOR_ITER(iw, w)
        {
            auto b = imbgr[ih][iw][0];
            auto g = imbgr[ih][iw][1];
            auto r = imbgr[ih][iw][2];
            auto minval = min(b, g);
            minval = min(minval, r);
            dc[ih][iw] = minval;
        }
    }
    vector<vector<double>> dark = Erode(dc, sz); // 进行腐蚀操作
    return dark;
}

// def AtmLight(im,dark):
//     [h,w] = im.shape[:2] ##cv2.imread()读取的彩色图像默认是BGR格式。
//     imsz = h*w
//     numpx = int(max(math.floor(imsz/1000),1))
//     darkvec = dark.reshape(imsz);
//     imvec = im.reshape(imsz,3);
//
//     indices = darkvec.argsort();
//     indices = indices[imsz-numpx::]### 在Python中，切片的一般形式是 [start:stop:step]。
//
//     atmsum = np.zeros([1,3])
//     for ind in range(1,numpx):
//     atmsum = atmsum + imvec[indices[ind]]
//
//     A = atmsum / numpx;
//     return A
//  Helper function to reshape 2D array into 1D
vector<int> ReshapeTo1D(const vector<vector<int>> &mat)
{
    vector<int> reshaped;
    for (const auto &row : mat)
    {
        reshaped.insert(reshaped.end(), row.begin(), row.end());
    }
    return reshaped;
}

// Helper function to reshape 3D array into 1D (for RGB channels)
template <class T>
vector<vector<T>> ReshapeTo1D(const vector<vector<vector<T>>> &im)
{
    vector<vector<T>> reshaped;
    for (const auto &row : im)
    {
        for (const auto &pixel : row)
        {
            reshaped.push_back(pixel);
        }
    }
    return reshaped;
}
template <class T>
vector<T> ReshapeTo1D(const vector<vector<T>> &im)
{
    vector<T> reshaped;
    for (const auto &row : im)
    {
        for (const auto &pixel : row)
        {
            reshaped.push_back(pixel);
        }
    }
    return reshaped;
}

// Custom argsort function: returns sorted indices based on values
template <class T>
vector<int> ArgSort(const vector<T> &vec)
{
    vector<int> indices(vec.size());
    //    iota(indices.begin(), indices.end(), 0);  // Create a vector of indices [0, 1, 2, ..., n-1]
    for (int i = 0; i < indices.size(); i++)
        indices[i] = i;
    // sort(indices.begin(), indices.end());
    sort(indices.begin(), indices.end(), [&](T a, T b)
    { return vec[a] < vec[b]; });
    return indices;
}

// AtmLight function in C++
vector<double> AtmLight(const vector<vector<vector<double>>> &im, const vector<vector<double>> &dark)
{
    int h = im.size();
    int w = im[0].size();
    int imsz = h * w;

    // Calculate number of pixels to consider
    int numpx = max(static_cast<int>(floor(imsz / 1000.0)), 1);

    // Reshape dark and image to 1D arrays
    auto darkvec = ReshapeTo1D(dark);
    auto imvec = ReshapeTo1D(im);

    // Sort indices based on the dark channel values
    auto indices = ArgSort(darkvec);

    // Select top `numpx` brightest points from the dark channel
    vector<vector<double>> brightestPixels;
    for (int i = imsz - numpx; i < imsz; ++i)
    {
        brightestPixels.push_back(imvec[indices[i]]);
    }

    // Sum up the RGB values of the brightest pixels
    vector<double> atmsum(3, 0.0);
    for (const auto &pixel : brightestPixels)
    {
        atmsum[0] += pixel[0]; // Blue channel
        atmsum[1] += pixel[1]; // Green channel
        atmsum[2] += pixel[2]; // Red channel
    }

    // Calculate the atmospheric light by averaging the sum
    vector<double> A(3);
    A[0] = atmsum[0] / numpx;
    A[1] = atmsum[1] / numpx;
    A[2] = atmsum[2] / numpx;

    return A;
}

// def TransmissionEstimate(im,A,sz):
//     omega = 0.95;
//     im3 = np.empty(im.shape,im.dtype);
//
//     for ind in range(0,3):
//     im3[:,:,ind] = im[:,:,ind]/A[0,ind]
//
//     transmission = 1 - omega*DarkChannel(im3,sz);
//     return transmission
//  TransmissionEstimate 函数
vector<vector<double>> TransmissionEstimate(const vector<vector<vector<double>>> &im, const vector<double> &A, int sz)
{
    double omega = 0.95;
    int rows = im.size();
    int cols = im[0].size();

    // 创建 im3，用于存储归一化后的图像
    vector<vector<vector<double>>> im3(rows, vector<vector<double>>(cols, vector<double>(3, 0.0)));

    // 按照每个通道对像素进行归一化
    for (int ind = 0; ind < 3; ++ind)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                im3[i][j][ind] = static_cast<double>(im[i][j][ind]) / A[ind];
            }
        }
    }

    // 将归一化后的图像传给 DarkChannel 函数，得到暗通道图
    vector<vector<double>> dark = DarkChannel(im3, sz);

    // 计算传输率图 (transmission)
    vector<vector<double>> transmission(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            transmission[i][j] = 1.0 - omega * static_cast<double>(dark[i][j]) / 255.0;
        }
    }

    return transmission;
}

// def Guidedfilter(im,p,r,eps):
//     mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
//     mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
//     mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
//     cov_Ip = mean_Ip - mean_I*mean_p;
//
//     mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
//     var_I   = mean_II - mean_I*mean_I;
//
//     a = cov_Ip/(var_I + eps);
//     b = mean_p - a*mean_I;
//
//     mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
//     mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));
//
//     q = mean_a*im + mean_b;
//     return q;
//  Helper function to apply a box filter (mean filter) with window size (r x r)
vector<vector<double>> BoxFilter(const vector<vector<double>> &im, int r)
{
    int rows = im.size();
    int cols = im[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0.0));

    // Apply box filter
    int kernel_size = 2 * r + 1;
    for (int i = r; i < rows - r; ++i)
    {
        for (int j = r; j < cols - r; ++j)
        {
            double sum = 0.0;
            for (int ki = -r; ki <= r; ++ki)
            {
                for (int kj = -r; kj <= r; ++kj)
                {
                    sum += im[i + ki][j + kj];
                }
            }
            result[i][j] = sum / (kernel_size * kernel_size);
        }
    }

    return result;
}

// Guidedfilter function in C++
vector<vector<double>> Guidedfilter(const vector<vector<double>> &I, const vector<vector<double>> &p, int r, double eps)
{
    int rows = I.size();
    int cols = I[0].size();

    // 1. Compute the mean of I, p and their products using box filter
    vector<vector<double>> mean_I = BoxFilter(I, r);
    vector<vector<double>> mean_p = BoxFilter(p, r);

    vector<vector<double>> Ip(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            Ip[i][j] = I[i][j] * p[i][j];
        }
    }
    vector<vector<double>> mean_Ip = BoxFilter(Ip, r);

    // 2. Compute the covariance of (I, p): cov_Ip = E(Ip) - E(I)E(p)
    vector<vector<double>> cov_Ip(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            cov_Ip[i][j] = mean_Ip[i][j] - mean_I[i][j] * mean_p[i][j];
        }
    }

    // 3. Compute the variance of I: var_I = E(II) - (E(I))^2
    vector<vector<double>> II(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            II[i][j] = I[i][j] * I[i][j];
        }
    }
    vector<vector<double>> mean_II = BoxFilter(II, r);

    vector<vector<double>> var_I(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            var_I[i][j] = mean_II[i][j] - mean_I[i][j] * mean_I[i][j];
        }
    }

    // 4. Compute the coefficients a and b
    vector<vector<double>> a(rows, vector<double>(cols, 0.0));
    vector<vector<double>> b(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            a[i][j] = cov_Ip[i][j] / (var_I[i][j] + eps);
            b[i][j] = mean_p[i][j] - a[i][j] * mean_I[i][j];
        }
    }

    // 5. Compute the mean of a and b using box filter
    vector<vector<double>> mean_a = BoxFilter(a, r);
    vector<vector<double>> mean_b = BoxFilter(b, r);

    // 6. Compute the output image q = mean_a * I + mean_b
    vector<vector<double>> q(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            q[i][j] = mean_a[i][j] * I[i][j] + mean_b[i][j];
        }
    }

    return q;
}

// def TransmissionRefine(im,et):
//     gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
//     gray = np.float64(gray)/255;
//     r = 60;
//     eps = 0.0001;
//     t = Guidedfilter(gray,et,r,eps);
//
//     return t;
//  将彩色图像转换为灰度图像
vector<vector<double>> ConvertToGray(const vector<vector<vector<int>>> &im)
{
    int rows = im.size();
    int cols = im[0].size();

    vector<vector<double>> gray(rows, vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // 使用加权平均法将 RGB 转换为灰度
            gray[i][j] = 0.299 * im[i][j][2] + 0.587 * im[i][j][1] + 0.114 * im[i][j][0];
        }
    }

    return gray;
}

// TransmissionRefine 函数
vector<vector<double>> TransmissionRefine(const vector<vector<vector<int>>> &im, const vector<vector<double>> &et)
{
    // Step 1: 将彩色图像转换为灰度图像
    vector<vector<double>> gray = ConvertToGray(im);

    // Step 2: 将灰度图像的像素值归一化到 [0, 1] 之间
    int rows = gray.size();
    int cols = gray[0].size();
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            gray[i][j] /= 255.0;
        }
    }

    // Step 3: 调用 Guidedfilter 函数进行引导滤波
    int r = 60;          // 半径
    double eps = 0.0001; // 正则化参数
    vector<vector<double>> t = Guidedfilter(gray, et, r, eps);

    return t;
}

// def Recover(im,t,A,tx = 0.1):
//     res = np.empty(im.shape,im.dtype);
//     t = cv2.max(t,tx);
//
//     for ind in range(0,3):
//     res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
//
//     return res
//  Recover函数实现
vector<vector<vector<double>>> Recover(const vector<vector<vector<double>>> &im, const vector<vector<double>> &t, const vector<double> &A, double tx = 0.1)
{
    int rows = im.size();
    int cols = im[0].size();
    int channels = 3; // 假设im为3通道图像

    // 初始化恢复后的结果图像
    vector<vector<vector<double>>> res(rows, vector<vector<double>>(cols, vector<double>(channels, 0.0)));

    // 限制 t 的最小值为 tx
    vector<vector<double>> t_max(rows, vector<double>(cols, tx));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            t_max[i][j] = max(t[i][j], tx);
        }
    }

    // 执行恢复操作: res[:,:,ind] = (im[:,:,ind] - A[0,ind]) / t + A[0,ind]
    for (int ind = 0; ind < channels; ++ind)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                res[i][j][ind] = (im[i][j][ind] - A[ind]) / t_max[i][j] + A[ind];
            }
        }
    }

    return res;
}

static int Dehaze(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;
    int pwl_idx = 0;

    int32_t *bgr_i = reinterpret_cast<int32_t *>(frame->data.bgr_s32_i);


    const DePwlPrms *pwl_prm = &(isp_prm->depwl_prm);

    vector<vector<vector<double>>> imbgr;
    vector<vector<vector<int>>> src;
    imbgr.resize(frame->info.height);
    FOR_ITER(ih, frame->info.height)
    {
        imbgr[ih].resize(frame->info.width);
        src[ih].resize(frame->info.width);
        FOR_ITER(iw, frame->info.width)
        {
            int pixel_idx = GET_PIXEL_INDEX(iw, ih, frame->info.width);

            auto b = bgr_i[3 * pixel_idx + 0];
            auto g = bgr_i[3 * pixel_idx + 1];
            auto r = bgr_i[3 * pixel_idx + 2];
            src[ih][iw]=vector<int>{b,g,r};
            imbgr[ih][iw] = vector<double>{b/ 255.0,g/ 255.0,r/ 255.0};
        }
    }
    //    src = cv2.imread(fn);
    //    I = src.astype('float64')/255;
    //
    //    dark = DarkChannel(I,15);
    //    A = AtmLight(I,dark);
    //    te = TransmissionEstimate(I,A,15);
    //    t = TransmissionRefine(src,te);
    //    J = Recover(I,t,A,0.1);

    auto &I = imbgr;
    auto dark = DarkChannel(I, 15);
    auto A = AtmLight(I, dark);
    auto te = TransmissionEstimate(I, A, 15);
    auto t = TransmissionRefine(src, te);
    auto J = Recover(I, t, A, 0.1);

    int32_t *bgr_o = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);
    FOR_ITER(ih, frame->info.height) {
        FOR_ITER(iw, frame->info.width) {
            int pixel_idx = GET_PIXEL_INDEX(iw, ih, frame->info.width);
            for(int k=0;k<3;k++){
                bgr_o[3 * pixel_idx + k] = static_cast<int32_t>(J[ih][iw][k]);
//                ClipMinMax<int32_t>(bgr_o[3 * pixel_idx + k], max_out_val, 0);
            }
        }
    }

    SwapMem<void>(frame->data.bgr_s32_i, frame->data.bgr_s32_o);

    return 0;
}

void RegisterDehazeMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::BGR;
    mod.out_domain = ColorDomains::BGR;

    mod.name = MOD_NAME;
    mod.run_function = Dehaze;

    RegisterIspModule(mod);
}