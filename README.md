# ACL Graph Add 替换

Host + Ascend C 混合 Add 示例。

接口：

```c
aclError add_replace_fp16(aclmdlRI modelRI, const char *exactName,
                          uint32_t *replacedCount);
```

函数遍历所有 Kernel。`exactName=NULL` 匹配 `Add_` 前缀，否则精确匹配。只修改 `funcHandle`、`numBlocks=24`，保留其他参数。

Kernel `add_replace_fp16_same_params` 计算 `y=x1+x2`。输入输出须为连续 `[24,1,128]`、float16；每 block 处理 128 个元素。当前不解析原始 Kernel 的 tilingdata，仅编写特化 Kernel；Shape 或数据类型变化后需重新匹配。当前该算子适配了sk入口。

`aclmdlRI` 系列接口为试验性接口，可能随版本变更，当前基于 CANN 9.2.0 验证，升级CANN版本后需重新检查兼容性，另外这一系列接口需在 capture 后、无并发访问时调用。

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

测试替换后调用 `graph.super_kernel_optimize()`，校验三轮 replay；910c/CANN 9.2.0通过。
