#include "easylogging++.h"
#include "EasyBMP.h"
#include "common/pipeline.h"

INITIALIZE_EASYLOGGINGPP

extern int ParseIspCfgFile(const std::string cfg_file_path, IspPrms &isp_prm);

int main()
{
    LOG(INFO) << "APP Start Running";
    IspPipeline pipeline;
    IspPrms isp_prms;

    char cfgFilePath[] = "./cfgs/isp_config_cannon.json";
    auto ret = ParseIspCfgFile(cfgFilePath, isp_prms);
    if (ret)
    {
        LOG(ERROR) << cfgFilePath << " parse failed\n";
        return -1;
    }
    auto width = isp_prms.info.width;
    auto height = isp_prms.info.height;
    Frame frame(isp_prms.info);

    frame.ReadFileToFrame(isp_prms.raw_file, width * height * isp_prms.info.bpp / 8);
    /*
    bpp (Bits Per Pixel):
这个字段表示每个像素的位深度，指每个像素所占用的位数。
'''
"info": {
        "sensor_name": "cannon",
        "cfa": "RGGB",
        "data_type": "RAW16",
        "bpp": 16,
        "max_bit": 14,
        "width": 6080,
        "height": 4044,
        "mipi_packed": 0
    },
'''
在你的配置文件中，bpp 设置为 16，表示每个像素占用 16 位。
这个值通常用于定义图像数据的精度。例如，RAW16 数据类型通常表示 16 位深度的原始图像数据。

mipi_packed:
这个字段表示数据是否采用了 MIPI (Mobile Industry Processor Interface) 协议的打包格式。
MIPI 是一种用于传输图像传感器数据的标准接口协议。mipi_packed 设置为 0 表示数据没有采用 MIPI 打包格式，而是以非打包的方式存储或传输。
mipi_packed 设置为 1 则表示数据采用了 MIPI 的打包格式，这种格式通常会将多个像素的数据压缩到一起，以减少带宽需求。
     */

    pipeline.MakePipe(isp_prms.pipe);
    pipeline.PrintPipe();

    ret = pipeline.RunPipe(&frame, &isp_prms);

    if (!ret)
    {
        std::string path = isp_prms.out_file_path+isp_prms.raw_file;

        // 找到最后一个斜杠的位置
        size_t lastSlash = path.find_last_of('/');
        // 找到最后一个点的位置
        size_t lastDot = path.find_last_of('.');
        std::string filenameWithoutExtension;
        if (lastSlash != std::string::npos && lastDot != std::string::npos && lastDot > lastSlash) {
            // 提取从斜杠后到点之前的部分
            filenameWithoutExtension = path.substr(lastSlash + 1, lastDot - lastSlash - 1);
            std::cout << "Filename without extension: " << filenameWithoutExtension << std::endl;
        } else {
            std::cout << "Invalid path format." << std::endl;
        }
        WriteBgrMemToBmp((isp_prms.out_file_path+filenameWithoutExtension + ".bmp").c_str(), (char *)frame.data.bgr_u8_o, width, height, 24);
        WriteMemToFile(isp_prms.out_file_path +filenameWithoutExtension+ "_bgr.raw", frame.data.bgr_u8_o, width * height * 3);
        LOG(INFO) << "APP Common Exit";
    }
    else
    {
        LOG(ERROR) << "Pipe run Error Exit";
    }

    return 0;
}
