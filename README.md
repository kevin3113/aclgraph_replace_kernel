# ACL Graph Add 替换

Host + Ascend C 混合示例，只处理固定规格 Add。

接口：

```c
aclError add_replace_fp16(aclmdlRI modelRI, const char *exactName,
                          uint32_t *replacedCount);
```

函数遍历所有 Kernel。`exactName=NULL` 匹配 `Add_` 前缀，否则精确匹配。只修改 `funcHandle`、`numBlocks=24`，保留其他参数；`taskGrp` 按 Runtime 要求设为 `NULL`，提交更新。

Kernel `add_replace_fp16_same_params` 计算 `y=x1+x2`。输入输出须为连续 `[24,1,128]`、float16；每 block 处理 128 个元素。当前不解析原始 Kernel 的 tilingdata，仅编写特化 Kernel；Shape 或数据类型变化后需重新匹配。另提供 `__sk__` 入口并用 `SK_BIND` 绑定普通入口。

调用须在 capture 结束且无并发访问时进行。`aclmdlRI` 系列接口当前为试验性接口，后续版本可能变更；本文基于 CANN 9.2.0 验证，升级 CANN 后需重新检查兼容性。

`test_add.py` 仅为测试，在 capture 期间通过 `aclmdlRICaptureGetInfo` 获取 `modelRI`。

## 编译与测试

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cmake -S add_replace_minimal -B build-add
cmake --build build-add -j4
```

```bash
python add_replace_minimal/test_add.py \
  --library build-add/libadd_replace.so --device 0
```

测试替换后调用 `graph.super_kernel_optimize()`，校验三轮 replay；910c/CANN 9.2.0 已通过。
