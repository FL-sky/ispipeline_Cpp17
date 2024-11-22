#include "modules/modules.h"

#define MOD_NAME "unpack"

static void UnpackRaw10ToRaw16(uint8_t *raw, uint16_t *unpack_raw16, int width, int height)
{
    uint8_t *raw10_packed_in = raw;
    uint16_t *raw16_unpacked_out = unpack_raw16;
    int pixel_idx = 0;

    for (int p1 = 0, p2 = 0; p2 < width * height; p1 += 5, p2 += 4)
    {
        raw10_packed_in = raw + p1;
        raw16_unpacked_out = unpack_raw16 + p2;
        raw16_unpacked_out[0] = ((uint16_t)raw10_packed_in[0] << 2) | (((uint16_t)raw10_packed_in[5] >> 6) & 0x03);
        raw16_unpacked_out[1] = ((uint16_t)raw10_packed_in[1] << 2) | (((uint16_t)raw10_packed_in[5] >> 4) & 0x03);
        raw16_unpacked_out[2] = ((uint16_t)raw10_packed_in[2] << 2) | (((uint16_t)raw10_packed_in[5] >> 2) & 0x03);
        raw16_unpacked_out[3] = ((uint16_t)raw10_packed_in[3] << 2) | (((uint16_t)raw10_packed_in[5] >> 0) & 0x03);
    }
}

static void UnpackRaw12ToRaw16(uint8_t *raw, uint16_t *unpack_raw16, int width, int height)
{
    uint8_t *raw12_packed_in = raw;
    uint16_t *raw16_unpacked_out = unpack_raw16;

    for (int p1 = 0, p2 = 0; p2 < width * height; p1 += 3, p2 += 2)
    {
        raw12_packed_in = raw + p1;
        raw16_unpacked_out = unpack_raw16 + p2;
        raw16_unpacked_out[0] = (raw12_packed_in[0] << 4) | ((raw12_packed_in[3] >> 4) & 0x0f); // 此处 raw12_packed_in[3] 应该为 raw12_packed_in[2]吧?
        raw16_unpacked_out[1] = (raw12_packed_in[1] << 4) | ((raw12_packed_in[3] >> 0) & 0x0f);
    }
}

static void UnpackRaw16ToRaw16(uint8_t *raw, uint16_t *unpack_raw16, int width, int height)
{
    uint8_t *raw16_packed_in = raw;
    uint16_t *raw16_unpacked_out = unpack_raw16;

    memcpy(raw16_unpacked_out, raw16_packed_in, width * height * 2);
}

static void UnpackRaw8ToRaw16(uint8_t *raw, uint16_t *unpack_raw16, int width, int height)
{
    uint8_t *raw16_packed_in = raw;
    uint16_t *raw16_unpacked_out = unpack_raw16;
    for (int i = 0; i < width * height; i++)
    {
        raw16_unpacked_out[i] = raw16_packed_in[i];
    }
    //    memcpy(raw16_unpacked_out, raw16_packed_in, width * height * 2);
}

static int MipiDataUnpack(Frame *frame, const IspPrms *prms)
{
    if ((frame == nullptr) || (prms == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    switch (frame->info.dt)
    {
    case RawDataTypes::RAW10:
        UnpackRaw10ToRaw16((uint8_t *)frame->data.raw_u8_i, (uint16_t *)frame->data.raw_u16_o, frame->info.width, frame->info.height);
        break;
    case RawDataTypes::RAW12:
        UnpackRaw12ToRaw16((uint8_t *)frame->data.raw_u8_i, (uint16_t *)frame->data.raw_u16_o, frame->info.width, frame->info.height);
        break;
    case RawDataTypes::RAW16:
        UnpackRaw16ToRaw16((uint8_t *)frame->data.raw_u8_i, (uint16_t *)frame->data.raw_u16_o, frame->info.width, frame->info.height);
        break;
    case RawDataTypes::RAW8:
        UnpackRaw8ToRaw16((uint8_t *)frame->data.raw_u8_i, (uint16_t *)frame->data.raw_u16_o, frame->info.width, frame->info.height);
    default:
        break;
    }

    SwapMem<void>(frame->data.raw_u16_o, frame->data.raw_u16_i);

    return 0;
}

void RegisterUnpackMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_UINT8;
    mod.out_type = DataPtrTypes::TYPE_UINT16;

    mod.in_domain = ColorDomains::RAW;
    mod.out_domain = ColorDomains::RAW;

    mod.name = MOD_NAME;

    mod.run_function = MipiDataUnpack;

    RegisterIspModule(mod);
}