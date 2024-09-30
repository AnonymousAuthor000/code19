# Code 19
Code for the submission 19. 

Note that this is a prototype tool, it does not support some operators or data types. It also does not support some optimizations of TFLite (e.g., XNNPACK for ARM&x86). 

## Download the code:

```
git clone https://github.com/zhoumingyi/code19.git
cd CustomDLCoder
```

## Build the dependency:

To create the conda environment:

```
conda env create -f code19.yaml
conda activate code19
```

Install the Flatbuffer:

```
conda install -c conda-forge flatbuffers
```

(if no npm) install the npm:

```
sudo apt-get install npm
```

Install the jsonrepair (https://github.com/josdejong/jsonrepair):

```
npm install -g jsonrepair
```

## Download the source code of the TensorFlow. Here we test our tool on v2.9.1.

```
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip
```

Unzip the file:

```
unzip v2.9.1
```
## (optional) Build TensorFlow python package from the source

In this part, we can build TensorFlow python package from the source. If we use the pre-build TF version, the extracted c code cannot perform the same as the pre-build TF version because the extracted c code is compiled by your own machine. It will cause inevitable convertion errors if you want to compare our tool with the pre-build TF python library. Note that before this step, you need to have a valid GCC environment in your environment.

Download the Bazel:

```
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

You can test the Bazel:

```
which bazel
```

It should return:

```
# in ubuntu
/usr/local/bin/bazel
```

Configure the build:

```
cp ./build_files/build.sh ./tensorflow-2.9.1/
cd tensorflow-2.9.1/
./configure
```

You can use the default setting (just type Return/Enter for every option).
Then build the TensorFlow python package from the source. Note that you can mofify the maximal number of jobs in the 'build.sh' script.

```
bash build.sh
cd ..
```

We rebuild the TF python package because the TFLite-cmake and our CustomDLCoder are compiled on your machine, rebuilding TF can help us to remove the error caused by compilation process and test different methods smoothly. If you have any problems in build TF from the source, please see: https://www.tensorflow.org/install/source .

## Copy the cmake script to the TFLite source project:  

```
cp -r ./build_files/coder ./tensorflow-2.9.1/tensorflow/lite/examples/
cp -r ./build_files/minimal ./tensorflow-2.9.1/tensorflow/lite/examples/
cp ./build_files/tflite_source/* ./tensorflow-2.9.1/tensorflow/lite/kernels/
```

<!-- ## Compile the baseline (cmake project of tflite models): 

In our paper, we compare our method with the original tflite cmake project (more details about tflite cmake: https://www.tensorflow.org/lite/guide/build_cmake). Note that to compare our method with baseline, we disable some optimizations because our method may not support them (we didn't do a comprehensive test). So you can compile the baseline:

```
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
``` -->

## Run DeepSign (when the signature is correct)

(1) In our paper, we compare our method with the original tflite cmake project (more details about tflite cmake: https://www.tensorflow.org/lite/guide/build_cmake). Note that to compare our method with baseline, we disable some optimizations because our method may not support them (we didn't do a comprehensive test). So you can compile the baseline:

```
cp -r ./build_files/minimal ./tensorflow-2.9.1/tensorflow/lite/examples/
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
```

Then, test our method on SqueezeNet model. You can also change the test model (e.g., set '--model_name=fruit' to use the fruit.tflite model in the 'tflite_model' folder). All models are compatible with the baseline (the provided tflite cmake project).

```
python main.py --model_name=squeezenet --verify_nb=5 --free_unused_data
```

If you want to maintain the intermediate data, you can unset the "--free_unused_data" (it will speed up the inference and saving the memory if it is True):

```
python main.py --model_name=squeezenet
```

The generated code can be found in *./tensorflow-2.9.1/tensorflow/lite/examples/coder*. The compiled shared library is in *./coder_x86_build/libcoder.so. We use the shared library as the output becasue it is easy to be tested by Python Code.*

(2) For teting our method on GPT-2, you need to first download the model through OneDrive (https://monashuni-my.sharepoint.com/:u:/g/personal/mingyi_zhou_monash_edu/EUq_riT5FVZClZZUUDlYDnkB5tT_j6YPtCmkUAPCvrMaFg?e=N8UkbV). The source model is collected from Hugginface (https://huggingface.co/distilgpt2).

Next, remove the exist cache files, copy the model to the './tflite_model' folder, and modify the tflite cmake file (the gpt2.tflite has different input type):

```
cd minimal_x86_build/ && rm -rf * &&cd ..
rm ./tensorflow-2.9.1/tensorflow/lite/examples/minimal/minimal.cc
mv ./tensorflow-2.9.1/tensorflow/lite/examples/minimal/minimal_gpt2.cc ./tensorflow-2.9.1/tensorflow/lite/examples/minimal/minimal.cc
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
```

Next, run:

```
python main.py --model_name=gpt2  --verify_nb=5 --free_unused_data
```

Note that testing on GPT2 needs a machine with large RAM (RAM size smaller than 64 Gb may cause termination of compilation). 
