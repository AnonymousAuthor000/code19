./coder_x86_build/shell 64 &
# ./minimal_x86_build/minimal ./tflite_model/fruit.tflite 150528
# ./your_program &
pid=$!
wait $pid
grep VmPeak /proc/$pid/status


# /usr/bin/time -v ./coder_x86_build/shell 64