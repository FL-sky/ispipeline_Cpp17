#include "modules/modules.h"
#include <map>

static std::map<std::string, IspModule> s_isp_mode_map;

int RegisterIspModule(IspModule mod)
{
    if (mod.name.empty())
    {
        return -1;
    }
    s_isp_mode_map[mod.name] = mod;
    return 0;
}

int GetIspModuleFromName(std::string name, IspModule &mod)
{
    if (s_isp_mode_map.count(name) > 0)
    {
        mod = s_isp_mode_map[name];
        return 0;
    }
    return -1;
}

int ShowAllIspModules()
{
    LOG(INFO) << "============== default pipeline mods show start ============";
    for (auto iter = s_isp_mode_map.rbegin(); iter != s_isp_mode_map.rend(); ++iter)
    {
        LOG(INFO) << "isp module-> [" << iter->first << "\t] | in bits ["
                  << std::to_string((int)iter->second.in_type) << "] | out bits [" << std::to_string((int)iter->second.out_type) << "]";
    }
    LOG(INFO) << "============== default pipeline mods show end ============";
    return 0;
}