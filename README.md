# ACL Graph Add 替换

Host + Ascend C 混合示例，只处理固定规格 Add。

接口：

```c
aclError add_replace_fp16(aclmdlRI modelRI, const char *exactName,
                          uint32_t *replacedCount);
```

函数遍历所有 Kernel。`exactName=NULL` 匹配 `Add_` 前缀，否则精确匹配。只修改 `funcHandle`、`numBlocks=24`，保留其他参数；`taskGrp` 按 Runtime 要求设为 `NULL`，提交更新。

Kernel `add_replace_fp16_same_params` 计算 `y=x1+x2`。输入输出须为连续 `[24,1,128]`、float16；每 block 处理 128 个元素。句柄通过 `aclrtGetFuncBySymbol` 获取，无需加载 Device 二进制。

调用须在 capture 结束且无并发访问时进行。

`test_add.py` 仅为测试，在 capture 期间通过 `aclmdlRICaptureGetInfo` 获取 `modelRI`。正式代码应由框架或 NPUGraph 接口提供句柄。

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

CMake 使用 `find_package(ASC)`、`LANGUAGES ASC CXX`、`LANGUAGE ASC`。测试校验三轮 replay 的 3072 个元素；910c 已通过。
