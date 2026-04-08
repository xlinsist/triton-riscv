// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: i32, %arg3: i32) {
    %in_ptrs = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>

    %x = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c2 = arith.constant 2 : i32
    %c2s = tt.splat %c2 : i32 -> tensor<128xi32>
    %x2 = arith.addi %x, %c2s : tensor<128xi32>
    %x2e = tt.expand_dims %x2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %xb = tt.broadcast %x2e : tensor<128x1xi32> -> tensor<128x256xi32>

    %y = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %c3 = arith.constant 3 : i32
    %c3s = tt.splat %c3 : i32 -> tensor<256xi32>
    %y3 = arith.addi %y, %c3s : tensor<256xi32>
    %c1024 = arith.constant 1024 : i32
    %c1024s = tt.splat %c1024 : i32 -> tensor<256xi32>
    %y3s = arith.muli %y3, %c1024s : tensor<256xi32>
    %y3e = tt.expand_dims %y3s {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %yb = tt.broadcast %y3e : tensor<1x256xi32> -> tensor<128x256xi32>

    %idx = arith.addi %xb, %yb : tensor<128x256xi32>
    %ldptr = tt.addptr %in_ptrs, %idx : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %stptr = tt.addptr %out_ptrs, %idx : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>

    %other = arith.constant 0xFF80 : bf16
    %arg2s = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mx = arith.cmpi slt, %x2, %arg2s : tensor<128xi32>
    %mxe = tt.expand_dims %mx {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %mxb = tt.broadcast %mxe : tensor<128x1xi1> -> tensor<128x256xi1>
    %arg3s = tt.splat %arg3 : i32 -> tensor<256xi32>
    %my = arith.cmpi slt, %y3, %arg3s : tensor<256xi32>
    %mye = tt.expand_dims %my {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %myb = tt.broadcast %mye : tensor<1x256xi1> -> tensor<128x256xi1>
    %mask = arith.andi %mxb, %myb : tensor<128x256xi1>

    %vals = tt.load %ldptr, %mask, %other : tensor<128x256x!tt.ptr<bf16>>
    tt.store %stptr, %vals, %mask : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel
// CHECK:      %[[IN_SUB:.*]] = memref.subview
// CHECK:      %[[IN_T:.*]] = bufferization.to_tensor %[[IN_SUB]]
// CHECK:      %[[INIT:.*]] = tensor.empty
// CHECK:      %[[FILLED:.*]] = linalg.fill
// CHECK:      %[[MERGED:.*]] = tensor.insert_slice %[[IN_T]] into %[[FILLED]]
// CHECK:      bufferization.materialize_in_destination
// CHECK-NOT:  memref.alloc
// CHECK-NOT:  memref.copy
