"""Minimal test for the fixed-shape FP16 Add replacement."""

import argparse
import ctypes
from pathlib import Path

import torch
import torch_npu


def check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed: aclError={ret}")


def configure_acl(acl):
    acl.aclmdlRICaptureGetInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
    acl.aclmdlRICaptureGetInfo.restype = ctypes.c_int


def capture(device):
    torch_npu.npu.set_device(device)
    stream = torch_npu.npu.Stream()
    status = ctypes.c_int()
    model = ctypes.c_void_p()
    host_x1 = torch.arange(3072).reshape(24, 1, 128).to(torch.float16) / 8
    host_x2 = torch.arange(3072).reshape(24, 1, 128).to(torch.float16) / 4
    with torch_npu.npu.stream(stream):
        x1 = host_x1.npu()
        x2 = host_x2.npu()
        graph = torch_npu.npu.NPUGraph()
        graph.capture_begin()
        try:
            output = x1 + x2
            acl = ctypes.CDLL("libascendcl.so")
            configure_acl(acl)
            check(acl.aclmdlRICaptureGetInfo(ctypes.c_void_p(stream.npu_stream), ctypes.byref(status), ctypes.byref(model)), "capture info")
        finally:
            graph.capture_end()
    if status.value != 1 or not model.value:
        raise RuntimeError("capture did not return an active modelRI")
    torch_npu.npu.current_stream().wait_stream(stream)
    torch_npu.npu.synchronize()
    return acl, graph, model, x1, x2, output, host_x1, host_x2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, required=True)
    options = parser.parse_args()
    acl, graph, model, x1, x2, output, host_x1, host_x2 = capture(options.device)
    library = ctypes.CDLL(str(options.library.resolve()))
    replace = library.add_replace_fp16
    replace.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
    replace.restype = ctypes.c_int
    try:
        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(output.cpu(), host_x1 + host_x2, rtol=0, atol=0)
        replaced = ctypes.c_uint32()
        check(replace(model, None, ctypes.byref(replaced)), "Add replacement")
        if replaced.value != 1:
            raise RuntimeError(f"Expected one Add task, replaced {replaced.value}")
        print("PASS: Add replacement committed")
        for index in range(3):
            current_x1 = host_x1 + index / 2
            current_x2 = host_x2 - index / 4
            x1.copy_(current_x1)
            x2.copy_(current_x2)
            output.fill_(float("nan"))
            torch_npu.npu.synchronize()
            graph.replay()
            torch_npu.npu.synchronize()
            torch.testing.assert_close(output.cpu(), current_x1 + current_x2, rtol=0, atol=0)
            print(f"PASS: replay {index + 1}, 3072 float16 elements")
    finally:
        torch_npu.npu.synchronize()
        graph.reset()


if __name__ == "__main__":
    main()
