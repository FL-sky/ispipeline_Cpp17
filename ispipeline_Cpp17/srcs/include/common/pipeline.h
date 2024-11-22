#ifndef ISP_PIPE_H
#define ISP_PIPE_H

#include <list>
#include <functional>

#include "modules/modules.h"
#include <chrono>

class IspPipeline
{
private:
    /* data */
    std::list<IspModule> pipe_;
    bool is_pipe_vaild_ = false;

public:
    IspPipeline();
    ~IspPipeline();

    IspPipeline(std::list<std::string> pipeline);

    int MakePipe(const std::list<std::string> &pipeline_str);
    int RunPipe(Frame *frame, const IspPrms *prms);
    int PrintPipe();
};

#endif // ! ISP_PIPE_H