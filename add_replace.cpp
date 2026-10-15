#include "kernel_operator.h"
#include <acl/acl_rt.h>
#include <array>
#include <cstring>
#include <vector>

struct AddReplaceArgs {
    GM_ADDR x1;
    GM_ADDR x2;
    GM_ADDR y;
};

__aicore__ inline void add_replace_body(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t block_idx)
{
    constexpr uint32_t n = 128;
    uint32_t off = block_idx * n;
    AscendC::TPipe p;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> q1, q2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qo;
    p.InitBuffer(q1, 1, n * sizeof(half)); p.InitBuffer(q2, 1, n * sizeof(half)); p.InitBuffer(qo, 1, n * sizeof(half));
    AscendC::GlobalTensor<half> g1, g2, go;
    g1.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x1) + off, n);
    g2.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x2) + off, n);
    go.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y) + off, n);
    auto a = q1.AllocTensor<half>(); auto b = q2.AllocTensor<half>();
    AscendC::DataCopy(a, g1, n); AscendC::DataCopy(b, g2, n); q1.EnQue(a); q2.EnQue(b);
    a = q1.DeQue<half>(); b = q2.DeQue<half>(); auto o = qo.AllocTensor<half>();
    AscendC::Add(o, a, b, n); qo.EnQue(o); q1.FreeTensor(a); q2.FreeTensor(b);
    o = qo.DeQue<half>(); AscendC::DataCopy(go, o, n); qo.FreeTensor(o);
    p.DestroyWithoutPipeAll();
}

extern "C" __global__ __vector__ __aicore__ void add_replace_fp16_same_params(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
{
    add_replace_body(x1, x2, y, AscendC::GetBlockIdx());
}

template <uint32_t split_num>
__sk__ __vector__ void add_replace_fp16_same_params_sk(
    const AddReplaceArgs *args, sk::SkSystemArgs *sys_args)
{
    add_replace_body(args->x1, args->x2, args->y, sys_args->SkGetBlockIdx());
}

SK_BIND(add_replace_fp16_same_params, 4,
        add_replace_fp16_same_params_sk<0>, add_replace_fp16_same_params_sk<1>,
        add_replace_fp16_same_params_sk<2>, add_replace_fp16_same_params_sk<3>);
namespace {
constexpr size_t kBytes = 3 * sizeof(void *);
thread_local int gLastError = ACL_SUCCESS;
aclError Error(aclError e) { gLastError = e; return e; }
}

extern "C" aclError add_replace_last_error() { return gLastError; }

extern "C" aclError add_replace_fp16(aclmdlRI model, const char *exactName, uint32_t *count)
{
    if (count) *count = 0;
    if (!model) return Error(ACL_ERROR_INVALID_PARAM);
    uint32_t ns = 0;
    aclError ret = aclmdlRIGetStreams(model, nullptr, &ns);
    if (ret) return Error(ret);
    std::vector<aclrtStream> streams(ns);
    ret = aclmdlRIGetStreams(model, streams.data(), &ns);
    if (ret) return Error(ret);
    struct Item { aclmdlRITask task; aclmdlRITaskParams params; };
    std::vector<Item> items;
    for (aclrtStream stream : streams) {
        uint32_t nt = 0;
        ret = aclmdlRIGetTasksByStream(stream, nullptr, &nt); if (ret) return Error(ret);
        std::vector<aclmdlRITask> tasks(nt);
        ret = aclmdlRIGetTasksByStream(stream, tasks.data(), &nt); if (ret) return Error(ret);
        for (aclmdlRITask task : tasks) {
            aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
            ret = aclmdlRITaskGetType(task, &type); if (ret) return Error(ret);
            if (type != ACL_MODEL_RI_TASK_KERNEL) continue;
            aclmdlRITaskParams params{};
            ret = aclmdlRITaskGetParams(task, &params); if (ret) return Error(ret);
            char name[65536]{};
            ret = aclrtGetFunctionName(params.kernelTaskParams.funcHandle, sizeof(name), name);
            if (ret) return Error(ret);
            bool match = exactName ? std::strcmp(name, exactName) == 0 : std::strncmp(name, "Add_", 4) == 0;
            if (!match) continue;
            if (params.kernelTaskParams.args == nullptr || params.kernelTaskParams.argsSize < kBytes ||
                params.kernelTaskParams.isHostArgs > 1) return Error(ACL_ERROR_INVALID_PARAM);
            Item item{task, params};
            item.params.taskGrp = nullptr;
            items.push_back(item);
        }
    }
    if (items.empty()) return Error(ACL_ERROR_KERNEL_NOT_FOUND);
    aclrtFuncHandle function = nullptr;
    ret = aclrtGetFuncBySymbol(reinterpret_cast<const void *>(&add_replace_fp16_same_params), &function);
    if (ret || !function) return Error(ret ? ret : ACL_ERROR_INVALID_RESOURCE_HANDLE);
    for (auto &item : items) {
        // Preserve every captured parameter except the function handle and the
        // launch block count. The original args/tiling/cfg remain untouched.
        item.params.kernelTaskParams.funcHandle = function;
        item.params.kernelTaskParams.numBlocks = 24;
        ret = aclmdlRITaskSetParams(item.task, &item.params); if (ret) return Error(ret);
    }
    ret = aclmdlRIUpdate(model); if (ret) return Error(ret);
    if (count) *count = static_cast<uint32_t>(items.size());
    return Error(ACL_SUCCESS);
}
