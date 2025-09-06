#!/bin/bash
git clone https://github.com/ggml-org/llama.cpp.git || echo "Repo déjà cloné"
cd llama.cpp || exit
cmake -B build -S . -DLLAMA_CUBLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
echo "Compilation terminée. LIBLLAMA sera : $(pwd)/build/libllama.dylib"
ls build/libllama.dylib
export LIBLLAMA=/Users/victorcarre/Code/Projects/llm-assistant/resources/models/llama.cpp/build/libllama.dylib
