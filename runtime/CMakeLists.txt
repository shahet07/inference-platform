add_library(runtime src/onnx_model.cpp)
target_include_directories(runtime PUBLIC include)
target_link_libraries(runtime PRIVATE onnxruntime::onnxruntime)
