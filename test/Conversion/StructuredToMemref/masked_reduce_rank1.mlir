// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func @masked_sum_rank1(%in : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.splat %n : i32 -> tensor<128xi32>
    %4 = arith.cmpi slt, %0, %3 : tensor<128xi32>
    %5 = tt.splat %cst : f32 -> tensor<128xf32>
    %6 = tt.load %2, %4, %5 : tensor<128x!tt.ptr<f32>>
    %7 = "tt.reduce"(%6) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %8 = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %8 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    tt.store %out, %7 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_sum_rank1
// CHECK: vector.load
// CHECK-NOT: memref.alloc() : memref<128xf32>
// CHECK-NOT: bufferization.to_tensor

// -----

module {
  tt.func @masked_max_rank1(%in : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.splat %n : i32 -> tensor<128xi32>
    %4 = arith.cmpi slt, %0, %3 : tensor<128xi32>
    %5 = tt.splat %cst : f32 -> tensor<128xf32>
    %6 = tt.load %2, %4, %5 : tensor<128x!tt.ptr<f32>>
    %7 = "tt.reduce"(%6) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %8 = arith.maxnumf %arg0, %arg1 : f32
      tt.reduce.return %8 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    tt.store %out, %7 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_max_rank1
// CHECK: vector.load
// CHECK-NOT: memref.alloc() : memref<128xf32>
// CHECK-NOT: bufferization.to_tensor

// -----

module {
  tt.func @masked_mul_rank1_fallback(%in : !tt.ptr<f32>, %out : !tt.ptr<f32>, %n : i32) {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.splat %n : i32 -> tensor<128xi32>
    %4 = arith.cmpi slt, %0, %3 : tensor<128xi32>
    %5 = tt.splat %cst : f32 -> tensor<128xf32>
    %6 = tt.load %2, %4, %5 : tensor<128x!tt.ptr<f32>>
    %7 = "tt.reduce"(%6) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %8 = arith.mulf %arg0, %arg1 : f32
      tt.reduce.return %8 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    tt.store %out, %7 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_mul_rank1_fallback
// CHECK: memref.alloc() : memref<128xf32>
// CHECK: bufferization.to_tensor

// -----

module {
  tt.func @masked_sum_rank2_fallback(%in : !tt.ptr<f32>, %out : !tt.ptr<f32>, %rows : i32, %cols : i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x8xi32>
    %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %5 = tt.broadcast %4 : tensor<1x8xi32> -> tensor<4x8xi32>
    %6 = tt.splat %cols : i32 -> tensor<4x8xi32>
    %7 = arith.cmpi slt, %5, %6 : tensor<4x8xi32>
    %8 = arith.muli %2, %cols : tensor<4x8xi32>
    %9 = arith.addi %8, %5 : tensor<4x8xi32>
    %10 = tt.splat %in : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %12 = tt.splat %cst : f32 -> tensor<4x8xf32>
    %13 = tt.load %11, %7, %12 : tensor<4x8x!tt.ptr<f32>>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %15 = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %15 : f32
    }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
    %16 = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    tt.store %16, %14 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @masked_sum_rank2_fallback
// CHECK: memref.alloc() : memref<4x8xf32>
// CHECK: bufferization.to_tensor
