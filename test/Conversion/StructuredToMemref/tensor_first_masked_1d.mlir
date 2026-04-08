// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: i32) {
    %in_ptrs = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %in_ptrs, %range : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %out_ptrs, %range : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %other_s = arith.constant 0xFF80 : bf16
    %other = tt.splat %other_s : bf16 -> tensor<128xbf16>
    %bound = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<128xi32>
    %vals = tt.load %ldptr, %mask, %other : tensor<128x!tt.ptr<bf16>>
    tt.store %stptr, %vals, %mask : tensor<128x!tt.ptr<bf16>>
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
