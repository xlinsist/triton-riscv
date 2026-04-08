// RUN: triton-shared-opt --triton-to-linalg-experimental="structured-ldst-mode=tensor-first-vector-cpu" %s | FileCheck %s

module {
  tt.func public @vector_add_tensor_first(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %a_s = tt.splat %a : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %b_s = tt.splat %b : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %c_s = tt.splat %c : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %pa = tt.addptr %a_s, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %pb = tt.addptr %b_s, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %pc = tt.addptr %c_s, %r : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %va = tt.load %pa : tensor<16x!tt.ptr<f32>>
    %vb = tt.load %pb : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %va, %vb : tensor<16xf32>
    tt.store %pc, %sum : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @vector_add_tensor_first
// CHECK:      %[[A:.*]] = memref.reinterpret_cast
// CHECK:      %[[A_SUB:.*]] = memref.subview %[[A]][0] [%{{.*}}] [1]
// CHECK:      %[[AT0:.*]] = bufferization.to_tensor %[[A_SUB]] restrict : memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %[[AT:.*]] = tensor.cast %[[AT0]]
// CHECK:      %[[B:.*]] = memref.reinterpret_cast
// CHECK:      %[[B_SUB:.*]] = memref.subview %[[B]][0] [%{{.*}}] [1]
// CHECK:      %[[BT0:.*]] = bufferization.to_tensor %[[B_SUB]] restrict : memref<?xf32, strided<[1], offset: ?>>
// CHECK:      %[[BT:.*]] = tensor.cast %[[BT0]]
// CHECK:      %[[SUM:.*]] = linalg.generic
// CHECK:      %[[C:.*]] = memref.reinterpret_cast
// CHECK:      %[[C_SUB:.*]] = memref.subview %[[C]][0] [%{{.*}}] [1]
// CHECK:      %[[SUM_CAST:.*]] = tensor.cast %[[SUM]]
// CHECK:      bufferization.materialize_in_destination %[[SUM_CAST]] in writable %[[C_SUB]]
// CHECK-NOT:  memref.alloc
// CHECK-NOT:  memref.copy
