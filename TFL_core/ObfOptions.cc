#include <stdlib.h>
#include <float.h>
namespace randomname {

float output_activation_min=-FLT_MAX;
float output_activation_max=FLT_MAX;

const int input_dims_size=;
const int32_t input_dims_raw=;

const int input_obf_dims_size=1;
const int32_t input_obf_dims_raw[1]={1};

float input_obf_raw=;
// float* input_v_obf=input_obf_raw;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;



auto* randomname(float* input_v, float mapped_hash) {

  // // tflite::ArithmeticParams op_params;

  // // SetActivationParams(output_activation_min, output_activation_max,      \
  // //                     &op_params);

  // // optimized_ops::BroadcastMulDispatch(op_params, RuntimeShape(input_dims_size, input_dims_raw),                        \
  // //           input_v, RuntimeShape(input_obf_dims_size, input_obf_dims_raw), \
  // //           &input_obf_raw, RuntimeShape(output_dims_size, output_dims_raw), \
  // //           output_data);


float* output_data = (float*)malloc(sizeof(float) * output_num);
float obf_scaled_value;
for (int i = 0; i < output_num; ++i) {
    obf_scaled_value = input_v[i] * input_obf_raw * mapped_hash;

    output_data[i] = (obf_scaled_value > output_activation_max)
                     ? output_activation_max
                     : ((obf_scaled_value < output_activation_min)
                        ? output_activation_min
                        : obf_scaled_value);
}

  return output_data;
}
}  // namespace randomname