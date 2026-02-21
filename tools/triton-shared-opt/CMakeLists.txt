get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

# Note: MLIRRegisterAllPasses and NVGPUIR are symbol libraries that may not exist
# MLIRRegisterAllPasses is provided by LLVM but may not be available in all builds
# NVGPUIR is only built when nvidia backend is enabled
# Allow undefined symbols for these optional libraries
target_link_options(triton-shared-opt PRIVATE -Wl,--allow-shlib-undefined)

mlir_check_all_link_libraries(triton-shared-opt)
