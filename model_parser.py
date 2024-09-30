import re,os
import random
import string
import fileinput
# import tensorflow as tf
import numpy as np
# import json
from utils.utils import *
from extract_conv import extract_conv
from extract_pool import extract_pool
from extract_fc import extract_fc
from extract_inout import extract_inout, extract_out, extract_in_obf
from extract_concat import extract_concat, extract_pack
from extract_split import extract_split
from extract_gather import extract_gather

tfl_source_path = './TFL_core/'
tfl_output_path = './TFL_out/'
# tf_source_path = './tf_source_file/'
# tf_output_path = './tf_output_file/'
tfl_build_path = './tensorflow-2.9.1/tensorflow/lite/examples/coder/'
# register_file = './tensorflow-2.9.1/tensorflow/lite/kernels/register.cc'
build_file = './tensorflow-2.9.1/tensorflow/lite/examples/coder/coder.cc'

def remove_dir(filepath, del_build=False):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path) and f != '.gitignore':
            os.remove(file_path)
            if del_build:
                os.remove(os.path.join(tfl_build_path, f))

def get_attributes_params(op, op_ori, interpreter, unknown_config, adv_value=1.0, act_obf=None, only_weights_value=1.0, only_bias_value=1.0, hash_deobf_value=1.0, hash_sign=False):
    kwargs = {}
    if op['builtin_options_type'] == 'Conv2DOptions' or op['builtin_options_type'] == 'DepthwiseConv2DOptions':
        kwargs, unknown_config = extract_conv(op_ori, kwargs, interpreter, unknown_config, adv_value, act_obf, only_weights_value, only_bias_value)

    elif op['builtin_options_type'] == 'AveragePool2DOptions' or op['builtin_options_type'] == 'MaxPool2DOptions':
        kwargs, unknown_config = extract_pool(op_ori, kwargs, interpreter, unknown_config)

    elif op['builtin_options_type'] == 'FullyConnectedOptions':
        kwargs, unknown_config = extract_fc(op_ori, kwargs, interpreter, unknown_config,  adv_value, act_obf, only_weights_value, only_bias_value)

    elif op['builtin_options_type'] == 'LogisticOptions':
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)

    elif op['builtin_options_type'] == 'SoftmaxOptions':
        try:
            beta = op['builtin_options']['beta']
        except:
            kwargs['beta='] = 'beta=1.0'
        else:
            kwargs['beta='] = 'beta=' + str(beta)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'ConcatenationOptions':
        try:
            axis = op['builtin_options']['axis']
        except:
            kwargs['axis='] = 'axis=3'
            # print("Warning: no axis of concat_op found")
        else:
            kwargs['axis='] = 'axis=' + str(axis)
        kwargs, unknown_config = extract_concat(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'ReshapeOptions':
        # try:
        #     keep_dims = op['builtin_options']['keep_dims']
        # except:
        #     kwargs['keep_dims='] = 'keep_dims=true'
        #     print("Warning: no pot_scale_int16 function found")
        # else:
        #     kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                shape_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze()
                shape_item_num = shape_tensor.size
                kwargs['shape='] = 'shape' + '{' + str(shape_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['shape_size='] = 'shape_size=' + str(shape_item_num)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'AddOptions':
        try:
            activation = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            # print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(activation)
        try:
            pot_scale_int16 = op['builtin_options']['pot_scale_int16']
        except:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=true'
            # print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=' + str.lower(str(pot_scale_int16))
        add_aug_num = 0
        input_arg = []
        for i in range(2):
            for tensor_details in interpreter.get_tensor_details():
                if tensor_details['index'] == op_ori['inputs'][i]:
                    kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                    kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                    try:
                        add_input = interpreter.get_tensor(tensor_details["index"])
                    except:
                        kwargs['input_' + str(i) + '_raw=;'] = ''
                        kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                        add_aug_num += 1
                        input_arg.append('float* input_v_' + str(i))
                    else:
                        if np.all(add_input == 0):
                            kwargs['input_' + str(i) + '_raw=;'] = ''
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                            add_aug_num += 1
                            input_arg.append('float* input_v_' + str(i))
                            # print('Warning: no input_' + str(i) + '_raw found')
                        else:
                            input_item_num = add_input.size
                            _, type_str = conv_data_type_parser(add_input.dtype)
                            kwargs['input_' + str(i) + '_raw='] = type_str + ' input_' + str(i) + '_raw[' + str(input_item_num) + ']=' + '{' + str(add_input.flatten('C').tolist()).strip('[').strip(']') + '}'
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw'] = type_str + '* input_v_' + str(i) + '=input_' + str(i) + '_raw'
        kwargs['auguments_placeholder'] = ', '.join(input_arg)
        kwargs, unknown_config = extract_out(op_ori, kwargs, interpreter, unknown_config)

    elif op['builtin_options_type'] == 'ReducerOptions':
        try:
            keep_dims = op['builtin_options']['keep_dims']
        except:
            kwargs['keep_dims='] = 'keep_dims=true'
            # print("Warning: no keep_dims of mean_op found")
        else:
            kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                axis_tensor = interpreter.get_tensor(tensor_details["index"])
                axis_item_num = axis_tensor.size
                kwargs['axis_input='] = 'axis_input[' + str(axis_item_num) + ']=' + '{' + str(axis_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['axis_size='] = 'axis_size=' + str(axis_item_num)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'ResizeBilinearOptions':
        try:
            align_corners = op['builtin_options']['align_corners']
        except:
            kwargs['align_corners='] = 'align_corners=false'
            # print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['align_corners='] = 'align_corners=' + str.lower(str(align_corners))
        try:
            half_pixel_centers = op['builtin_options']['half_pixel_centers']
        except:
            kwargs['half_pixel_centers='] = 'half_pixel_centers=true'
            # print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['half_pixel_centers='] = 'half_pixel_centers=' + str.lower(str(half_pixel_centers))
        try:
            new_width = op['builtin_options']['new_width']
        except:
            kwargs['new_width='] = 'new_width=0'
        else:
            kwargs['new_width='] = 'new_width=' + str(new_width)
        try:
            new_height = op['builtin_options']['new_height']
        except:
            kwargs['new_height='] = 'new_height=0'
        else:
            kwargs['new_height='] = 'new_height=' + str(new_height)
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                size_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze()
                size_item_num = size_tensor.size
                size_dims_raw = '{' + str(size_tensor.shape).strip('(').strip(')') + '}'
                size_dims_size = len(size_tensor.shape)
                kwargs['size_raw='] = 'size_raw[' + str(size_item_num) + ']=' + '{' + str(size_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['size_dims_size='] = 'size_dims_size=' + str(size_dims_size)
                kwargs['size_dims_raw='] = 'size_dims_raw[' + str(size_dims_size) + ']=' + size_dims_raw
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] in ['ReluOptions', 'TanhOptions']:
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'SqueezeOptions':
        try:
            squeeze_dims = op['builtin_options']['squeeze_dims']
        except:
            raise ValueError("no squeeze_dims found")
        else:
            kwargs['squeeze_dim='] = 'squeeze_dim[8]=' + '{' + str(squeeze_dims).strip('[').strip(']') + '}'
            kwargs['num_squeeze_dim='] = 'num_squeeze_dim=' + str(len(squeeze_dims))
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'GatherOptions':
        kwargs, unknown_config = extract_gather(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'MeanOptions':
        try:
            keep_dims = op['builtin_options']['keep_dims']
        except:
            kwargs['keep_dims='] = 'keep_dims=true'
        else:
            kwargs['keep_dims='] = 'keep_dims=' + str(keep_dims)
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                axis_tensor = interpreter.get_tensor(tensor_details["index"])
                axis_item_num = axis_tensor.size
                kwargs['axis='] = 'axis[' + str(axis_item_num) + ']=' + '{' + str(axis_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['num_axis='] = 'num_axis=' + str(axis_item_num)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] in ['SquaredDifferenceOptions','PowOptions']:
        add_aug_num = 0
        input_arg = []
        for i in range(2):
            for tensor_details in interpreter.get_tensor_details():
                if tensor_details['index'] == op_ori['inputs'][i]:
                    # print(len(tensor_details['shape']))
                    # print(tuple(tensor_details['shape']))
                    if len(tensor_details['shape']) == 0:
                        kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[1]={1}'
                        kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=1'
                    else:
                        kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                        kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                    try:
                        add_input = interpreter.get_tensor(tensor_details["index"])
                    except:
                        kwargs['input_' + str(i) + '_raw=;'] = ''
                        kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                        add_aug_num += 1
                        input_arg.append('float* input_v_' + str(i))
                    else:
                        if np.all(add_input == 0):
                            kwargs['input_' + str(i) + '_raw=;'] = ''
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                            add_aug_num += 1
                            input_arg.append('float* input_v_' + str(i))
                            # print('Warning: no input_' + str(i) + '_raw found')
                        else:
                            input_item_num = add_input.size
                            _, type_str = conv_data_type_parser(add_input.dtype)
                            kwargs['input_' + str(i) + '_raw='] = type_str + ' input_' + str(i) + '_raw[' + str(input_item_num) + ']=' + '{' + str(add_input.flatten('C').tolist()).strip('[').strip(']') + '}'
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw'] = type_str + '* input_v_' + str(i) + '=input_' + str(i) + '_raw'
        kwargs['auguments_placeholder'] = ', '.join(input_arg)
        kwargs, unknown_config = extract_out(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'RsqrtOptions':
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] in ['SubOptions', 'MulOptions', 'DivOptions']:
        try:
            pot_scale_int16 = op['builtin_options']['pot_scale_int16']
        except:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=true'
            # print("Warning: no pot_scale_int16 function found")
        else:
            kwargs['pot_scale_int16='] = 'pot_scale_int16=' + str.lower(str(pot_scale_int16))
        try:
            fused_activation_function = op['builtin_options']['fused_activation_function']
        except:
            kwargs['activation='] = 'activation=kTfLiteActNone'
            # print("Warning: no activation function found")
        else:
            kwargs['activation='] = 'activation=' + conv_activation_parser(fused_activation_function)
        add_aug_num = 0
        input_arg = []
        for i in range(2):
            for tensor_details in interpreter.get_tensor_details():
                if tensor_details['index'] == op_ori['inputs'][i]:
                    kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                    kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                    try:
                        add_input = interpreter.get_tensor(tensor_details["index"])
                    except:
                        kwargs['input_' + str(i) + '_raw=;'] = ''
                        kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                        add_aug_num += 1
                        input_arg.append('float* input_v_' + str(i))
                    else:
                        if np.all(add_input == 0):
                            kwargs['input_' + str(i) + '_raw=;'] = ''
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw;'] = ''
                            add_aug_num += 1
                            input_arg.append('float* input_v_' + str(i))
                            # print('Warning: no input_' + str(i) + '_raw found')
                        else:
                            input_item_num = add_input.size
                            _, type_str = conv_data_type_parser(add_input.dtype)
                            kwargs['input_' + str(i) + '_raw='] = type_str + ' input_' + str(i) + '_raw[' + str(input_item_num) + ']=' + '{' + str(add_input.flatten('C').tolist()).strip('[').strip(']') + '}'
                            kwargs['input_v_' + str(i) + '=input_' + str(i) + '_raw'] = type_str + '* input_v_' + str(i) + '=input_' + str(i) + '_raw'
        kwargs['auguments_placeholder'] = ', '.join(input_arg)
        kwargs, unknown_config = extract_out(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'SplitOptions':
        try:
            num_splits = op['builtin_options']['num_splits']
        except:
            raise ValueError("no num_splits found")
        else:
            kwargs['num_splits='] = 'num_splits=' + str(num_splits)
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][0]:
                axis_tensor = interpreter.get_tensor(tensor_details["index"])
                axis_item_num = axis_tensor.size
                kwargs['axis='] = 'axis[' + str(axis_item_num) + ']=' + '{' + str(axis_tensor.tolist()).strip('[').strip(']') + '}'
                # kwargs['num_axis='] = 'num_axis=' + str(axis_item_num)
        kwargs, unknown_config = extract_split(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'TransposeOptions':
        # try:
        #     keep_dims = op['builtin_options']['keep_dims']
        # except:
        #     kwargs['keep_dims='] = 'keep_dims=true'
        #     print("Warning: no pot_scale_int16 function found")
        # else:
        #     kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                shape_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze()
                shape_item_num = shape_tensor.size
                kwargs['perm='] = 'perm=' + '{' + str(shape_tensor.tolist()).strip('[').strip(']') + '}'
                kwargs['perm_size='] = 'perm_size=' + str(shape_item_num)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'PackOptions':
        try:
            axis = op['builtin_options']['axis']
        except:
            # print("Warning: no axis found in Pack OP")
            kwargs['axis='] = 'axis=0'
            # raise ValueError("no axis found")
        else:
            kwargs['axis='] = 'axis=' + str(axis)
        try:
            values_count = op['builtin_options']['values_count']
        except:
            raise ValueError("no values_count found")
        else:
            kwargs['values_count='] = 'values_count=' + str(values_count)
        kwargs, unknown_config = extract_pack(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'SliceOptions':
        # try:
        #     keep_dims = op['builtin_options']['keep_dims']
        # except:
        #     kwargs['keep_dims='] = 'keep_dims=true'
        #     print("Warning: no pot_scale_int16 function found")
        # else:
        #     kwargs['keep_dims='] = 'keep_dims=' + str.lower(str(keep_dims))
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op_ori['inputs'][1]:
                begin_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze().tolist()
                begins_slice = [0,0,0,0,0]
                for i in range(len(begin_tensor)):
                    begins_slice[i+(len(begins_slice)-len(begin_tensor))] = begin_tensor[i]
                kwargs['begins='] = 'begins=' + '{' + str(begins_slice).strip('[').strip(']') + '}'
            elif tensor_details['index'] == op_ori['inputs'][2]:
                size_tensor = interpreter.get_tensor(tensor_details["index"]).squeeze().tolist()
                # shape_item_num = shape_tensor.size
                sizes_slice = [1,1,1,1,1]
                for i in range(len(size_tensor)):
                    sizes_slice[i+(len(sizes_slice)-len(size_tensor))] = size_tensor[i]
                kwargs['sizes='] = 'sizes=' + '{' + str(sizes_slice).strip('[').strip(']') + '}'
                # kwargs['perm_size='] = 'perm_size=' + str(shape_item_num)
        kwargs, unknown_config = extract_inout(op_ori, kwargs, interpreter, unknown_config)
    elif op['builtin_options_type'] == 'ObfOptions':
        if act_obf == "RELU6":
            kwargs['output_activation_max=FLT_MAX'] = 'output_activation_max=6.0'
            kwargs['output_activation_min=-FLT_MAX'] = 'output_activation_min=0.0'
        elif act_obf == "RELU":
            kwargs['output_activation_max=FLT_MAX'] = 'output_activation_max=FLT_MAX'
            kwargs['output_activation_min=-FLT_MAX'] = 'output_activation_min=0.0'
        elif act_obf == None:
            kwargs['output_activation_max=FLT_MAX'] = 'output_activation_max=FLT_MAX'
            kwargs['output_activation_min=-FLT_MAX'] = 'output_activation_min=-FLT_MAX'
        else:
            raise ValueError("fused activation function not found, Op sign: " + str(op['sign']))
        try:
            fc_filter_obf = op['fc_filter_obf']
        except:
            fc_filter_adv_value = 1.0
        else:
            fc_filter_adv_value = fc_filter_obf

        if hash_sign:
            if (1.0 - abs((adv_value*fc_filter_adv_value*only_weights_value/hash_deobf_value)%1)) < 0.000001 or \
                (1.0 - abs((adv_value*fc_filter_adv_value*only_weights_value/hash_deobf_value)%1)) > 0.999999:
                kwargs['input_obf_raw='] = 'input_obf_raw=' + str(round(adv_value*fc_filter_adv_value*only_weights_value/hash_deobf_value))
            else:
                kwargs['input_obf_raw='] = 'input_obf_raw=' + str(adv_value*fc_filter_adv_value*only_weights_value/hash_deobf_value)
        else:
            kwargs['    obf_scaled_value = input_v[i] * input_obf_raw * mapped_hash;'] = '    obf_scaled_value = input_v[i] * input_obf_raw;'
            kwargs['input_obf_raw='] = 'input_obf_raw=' + str(adv_value*fc_filter_adv_value*only_weights_value)
        # kwargs['input_obf_raw='] = 'input_obf_raw=' + str(round(adv_value*fc_filter_adv_value*only_weights_value/hash_deobf_value))
        # kwargs['input_obf_raw='] = 'input_obf_raw=' + str(adv_value*fc_filter_adv_value*only_weights_value)
        kwargs, unknown_config = extract_in_obf(op, kwargs, interpreter, unknown_config)
    return kwargs

def code_generator(op, kwargs, tfl_filelist, input_details, jsontext, op_sign, inout_list):
    name_str = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
    jsontext['oplist'].append({'LayerID':name_str, 'OpName':op['builtin_options_type'], 'input': op['inputs'], 'output': op['outputs'], 'sign': op_sign})
    with open('./oplist.txt', "a", encoding="utf-8") as f:
        f.write(name_str + '.  ' + op['builtin_options_type'] + '\n')
    input_num = 0
    for i in op['inputs']:
        if i in inout_list:
            input_num += 1

    for i in range(len(tfl_filelist)):
        if op['builtin_options_type'] == os.path.splitext(tfl_filelist[i])[0]:
            # print(op['builtin_options_type'])

            with open(os.path.join(tfl_source_path,tfl_filelist[i]), "r", encoding="utf-8") as f1,open(os.path.join(tfl_output_path,("%s.cc" % name_str)), "w", encoding="utf-8") as f2:
                for line in f1:
                    find_key = False
                    for key in kwargs:
                        if key in line:
                            # if key == 'RuntimeShape(bias_dims_size,bias_dims_raw),bias_tensor_data,':
                            #     print('identified:', key)
                            #     print(kwargs[key])
                            #     print(line)
                            #     print(re.sub(re.escape(key),kwargs[key],line))
                            f2.write(re.sub('randomname',name_str, re.sub(re.escape(key),kwargs[key],line)))
                            del kwargs[key]
                            find_key = True
                            break
                    # if op_sign > 0 and 'return output_data;' in line:
                    #     if input_num > 1:
                    #         for i in range(input_num):
                    #             f2.write("  free(input_v_%s);\n" % str(i))
                    #     else:
                    #         f2.write("  free(input_v);\n")
                    if not find_key and 'randomname' in line:
                        f2.write(re.sub('randomname',name_str,line))
                    elif not find_key:
                        f2.write(line)
            os.system("cp %s " % (os.path.join(tfl_output_path,("%s.cc" % name_str))) + " %s" % (os.path.join(tfl_build_path,("%s.cc" % name_str))))

    # with fileinput.input(files=build_file, inplace=True) as f:
    #     for line in f:
    #         if 'add_cus_here' in line:
    #             print("    \"%s.cc\"," % name_str)
    #         print(line, end="")

    # os.system("bash ./tf_output_file/%s.sh" % name_str)

def del_previous_file(build_file):

    with fileinput.input(files=build_file, inplace=True) as f:
        del_sign = False
        for line in f:
            if 'end files' in line or 'end function' in line:
                del_sign = False
            if 'add files' in line or 'add function' in line:
                print(line, end="")
                del_sign = True
            if not del_sign:
                print(line, end="")
    remove_dir(tfl_output_path, del_build=True)

def correct_json(interpreter, model_json):
    op_details = interpreter._get_ops_details()
    json_op_details = model_json['subgraphs'][0]["operators"]
    for i in range(len(json_op_details)):
        # print(json_op_details[i])
        if 'builtin_options_type' in json_op_details[i]:
            if json_op_details[i]['builtin_options_type'] == 'Pool2DOptions':
                if op_details[i]['op_name'] == 'AVERAGE_POOL_2D':
                    json_op_details[i]['builtin_options_type'] = 'AveragePool2DOptions'
                elif op_details[i]['op_name'] == 'MAX_POOL_2D':
                    json_op_details[i]['builtin_options_type'] = 'MaxPool2DOptions'
        else:
            if interpreter._get_ops_details()[i]['op_name'] == 'RESHAPE':
                json_op_details[i]['builtin_options_type'] = 'ReshapeOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'LOGISTIC':
                json_op_details[i]['builtin_options_type'] = 'LogisticOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'RELU':
                json_op_details[i]['builtin_options_type'] = 'ReluOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'TANH':
                json_op_details[i]['builtin_options_type'] = 'TanhOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'RSQRT':
                json_op_details[i]['builtin_options_type'] = 'RsqrtOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
            elif interpreter._get_ops_details()[i]['op_name'] == 'SLICE':
                json_op_details[i]['builtin_options_type'] = 'SliceOptions'
            elif interpreter._get_ops_details()[i]['op_name'] == 'POW':
                json_op_details[i]['builtin_options_type'] = 'PowOptions'
                # print("Warning: didn't find the builtin_options_type for this op")
    # return json_op_details
    return model_json


def lib_generator(model_json, ori_model_json, interpreter, inout_list, injected_pair={},
                  adv_obf_value={}, activation_type={}, intermediate_nodes={}, hash_deobf_value=1.0):
    tfl_filelist = os.listdir(tfl_source_path)
    op_sign = 0
    file = open('./oplist.txt', 'w').close()
    del_previous_file(build_file)
    input_details = (interpreter.get_input_details())[0]['shape'].astype(np.int32).tolist()

    jsontext = {'oplist':[]}
    # json_op_details = correct_json(interpreter, model_json)
    json_op_details = model_json['subgraphs'][0]["operators"]
    unknown_config = []
    # print(json_op_details)
    for op in json_op_details:

        act_obf = None
        # print(op)
        adv_value = 1.0
        # only_bias = []
        only_bias_value = 1.0
        only_weights_value = 1.0

        hash_sign = False # for the end_node ObfOptions in the pair, apply the hash value

        # hash_value = 1.0 # for the ObfOptions without the transformation, do not apply the hash value
        if injected_pair.get(op['sign']):
            adv_value = adv_value * adv_obf_value[op['sign']]
        else:
            adv_value = 1.0

        for i in injected_pair.keys():
            for j in range(len(injected_pair[i])):
                if op['sign'] == injected_pair[i][j]:
                    # adv_value = adv_value * (1.0/adv_obf_value[i])
                    only_weights_value = only_weights_value * (1.0/adv_obf_value[i])
                    act_obf = activation_type[i][j]
                    if op['builtin_options_type'] == 'ObfOptions':
                        hash_sign = True
                    #     hash_value = hash_deobf_value
                    break

        # ------------------------------------------------------------------------
        # For the intermediate transformable Op, only needs to transform the bias
        # The detail can be found in the paper of DynaMO
        # ------------------------------------------------------------------------
        for i in intermediate_nodes.keys():
            for j in range(len(intermediate_nodes[i])):
                if op['sign'] in intermediate_nodes[i]:
                    only_bias_value = only_bias_value * adv_obf_value[i]
                    # only_bias = True
                    # act_obf = activation_type[i][j]
                    break

        # print("model_json op: ", op['builtin_options_type'])
        # print("adv_value: ", adv_value)
        # print("act_obf: ", act_obf)
        # print("only_bias_value: ", only_bias_value)

        # print(op)
        # ---------------------------------------------------------------------------------------
        # If the operatpr is not ObfOptions, then use the computational graph without obfuscation
        # This is because the ObfOptions do not has input/outputs details so the code cannot
        # create the input/output information of customized codes
        # ---------------------------------------------------------------------------------------
        if op['builtin_options_type'] == 'ObfOptions':
            op_ori = None
        else:
            op_ori = ori_model_json['subgraphs'][0]["operators"][op_sign]
        kwargs = get_attributes_params(op, op_ori, interpreter, unknown_config, adv_value, act_obf,
                                        only_weights_value, only_bias_value, hash_deobf_value, hash_sign)
        code_generator(op, kwargs, tfl_filelist, input_details, jsontext, op_sign, inout_list)
        # print(kwargs)
        op_sign = op_sign + 1
        # if op_sign == 1059:
        #     break
    return jsontext, unknown_config
