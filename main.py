import os
import sys
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import json
import random
import argparse
# import orjson
# from tensorflow import keras
from collections import deque
from model_assembler import model_assembler
from model_parser import *
from utils.utils import *
from dinamic_config import dinamic_config

# ------------------------------------------------
# New code
# ------------------------------------------------

trans_enabled_list = ["DepthwiseConv2DOptions", "Conv2DOptions", "FullyConnectedOptions", "ObfOptions"]
linear_list = ["DepthwiseConv2DOptions", "Conv2DOptions", "FullyConnectedOptions", "ObfOptions", "RESHAPE",
              "AveragePool2DOptions", 'MaxPool2DOptions', "SqueezeOptions", "AddOptions", "Pool2DOptions",
              "ReducerOptions", "ReshapeOptions", "ConcatenationOptions", "ResizeBilinearOptions", "PackOptions",
              "SubOptions", "DivOptions", "MulOptions", "SliceOptions", "GatherOptions", "MeansOptions",
              "SplitOptions", "TransposeOptions"]
nonlinear_list = ["ReluOptions", "Relu6Options", "TanhOptions", "LogisticOptions", "SoftmaxOptions", "RELU",
                  "ResizeBilinearOptions", "RsqrtOptions", "SquaredDifferenceOptions", "PowOptions"]
nonlinear_enable_list = ["ObfOptions"]

# ------------------------------------------------
# End New code
# ------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lenet', help='name of the model')
parser.add_argument('--free_unused_data', action='store_true', help='free unused intermediate data')
parser.add_argument('--executable', action='store_true', help='generate executable file')
parser.add_argument('--extra_layer', type=int, default=30, help='number of extra layers')
parser.add_argument('--shortcut', type=int, default=0, help='number of shortcuts')
parser.add_argument('--verify_nb', type=int, default=5, help='number of shortcuts')
parser.add_argument('--hash_deobf_value', type=float, default=1.74318, help='the mapped value of the hash value for deobfuscation')
opt = parser.parse_args()

# ------------------------------------------------
# New code
# ------------------------------------------------
def nodes_within_distance(edges_obf, start_node):
    # Initialize the queue with the starting node and a distance of 0
    queue = deque([(start_node, 0)])
    # Set to keep track of visited nodes
    visited = set()
    # List to store nodes within distance less than 3
    result = []

    while queue:
        current_node, current_distance = queue.popleft()

        # If current distance is less than 3, add the node to the result
        if current_distance < 4:
            result.append(current_node)

        # If current distance is 3 or more, we stop expanding that node
        if current_distance >= 4:
            continue

        # Mark the current node as visited
        visited.add(current_node)

        # Expand to the next level
        for neighbor in edges_obf.get(current_node, []):
            if neighbor not in visited:
                queue.append((neighbor, current_distance + 1))
                visited.add(neighbor)  # Mark as visited as soon as it is added to the queue

    # Remove the start_node from result if present
    if start_node in result:
        result.remove(start_node)

    return result

def find_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_paths(graph, node, end, path)
            for p in new_paths:
                paths.append(p)
    return paths

def find_all_paths(edges_obf, node_list, start_node):
    all_paths = []
    # for start_node in edges_obf.keys():
    for end_node in node_list:
        if start_node != end_node:
            paths = find_paths(edges_obf, start_node, end_node)
            all_paths.extend(paths)
    return all_paths

def check_nodes_in_paths(node1, node2, all_paths):
    # node1 = node_list[i]
    # all_paths = []
    # # for start_node in edges_obf.keys():
    # for end_node in node_list:
    #     if start_node != end_node:
    #         paths = find_paths(edges_obf, start_node, end_node)
    #         all_paths.extend(paths)
    for path in all_paths:
        if node1 in path and node2 in path:
            return True
    return False

# def choose_random_node(A, B):
#     # Filter nodes with label 1
#     nodes_with_label_1 = [node for node, label in zip(A, B) if label == 1]

#     # Randomly select one node from the filtered list
#     if nodes_with_label_1:
#         return random.choice(nodes_with_label_1)
#     else:
#         return None  # If no nodes have label 1, return None or handle as needed

def pick_random_node_with_label_one(D):
    # Filter the dictionary to get a list of nodes with label 1
    nodes_with_label_one = [node for node, label in D.items() if label == 1]

    # Check if there are any nodes with label 1
    if not nodes_with_label_one:
        return None

    # Randomly select one node from the list
    return random.choice(nodes_with_label_one)

def is_single_edge_destination(edges_obf, input_node):
    # Count how many times the input_node appears as a destination
    count = 0
    for destinations in edges_obf.values():
        if input_node in destinations:
            count += 1
        if count > 1:
            return False
    return True

def get_op_detail_by_sign(model_json, sign):
    for i in range(len(model_json['subgraphs'][0]["operators"])):
        if model_json['subgraphs'][0]["operators"][i]["sign"] == sign:
            return model_json['subgraphs'][0]["operators"][i]
        else:
            None

# ------------------------------------------------
# End New code
# ------------------------------------------------

def reduce_size_json(json_file):
    with fileinput.input(files=json_file, inplace=True) as f:
        keep_sign = True
        for line in f:
            if 'buffers:' in line:
                keep_sign = False
                print('}', end="")
            if keep_sign:
                print(line, end="")


def random_shortcut(model_json):
    num_op = len(model_json['subgraphs'][0]["operators"])
    for i in range(opt.shortcut):
        rand_shortcut_pair = random.sample(range(0, num_op), 2)
        rand_shortcut_pair.sort()
        op_i = model_json['subgraphs'][0]["operators"][rand_shortcut_pair[1]]
        if 'builtin_options_type' in op_i.keys():
            if op_i['builtin_options_type'] != 'ConcatenationOptions' and op_i['builtin_options_type'] != 'AddOptions':
        # if op_i['builtin_options_type'] != 'ConcatenationOptions':
                op_i["inputs"].append(model_json['subgraphs'][0]["operators"][rand_shortcut_pair[0]]["outputs"][0])


def random_extra(model_json, out_start_point):
    num_op = len(model_json['subgraphs'][0]["operators"])
    for i in range(opt.extra_layer):
        rand_extra_pair = []
        rand_extra_start = random.sample(range(0, num_op), 1)[0]
        # print(rand_extra_start)
        rand_extra_pair.append(rand_extra_start)
        end_tmp = []
        for i in range(len(model_json['subgraphs'][0]["operators"])):
            if model_json['subgraphs'][0]["operators"][rand_extra_start]["outputs"][0] in model_json['subgraphs'][0]["operators"][i]["inputs"]:
                end_tmp.append(i)
        # print(end_tmp)
        if end_tmp == []:
            continue
        rand_extra_end = random.sample(end_tmp, 1)[0]
        rand_extra_pair.append(rand_extra_end)
        rand_extra_pair.sort()
        op_i = model_json['subgraphs'][0]["operators"][rand_extra_pair[1]]
        if 'builtin_options_type' in op_i.keys():
            if op_i['builtin_options_type'] != 'ConcatenationOptions' and op_i['builtin_options_type'] != 'AddOptions':
                # op_i["inputs"].insert(0, out_start_point+i)
                # ------------------------------------------------------------------------
                # If the start point is the obf layer, we need to add the ori_input info
                # to let the codes generate correct input details of customized code.
                # ------------------------------------------------------------------------
                if 'builtin_options_type' in model_json['subgraphs'][0]["operators"][rand_extra_pair[0]].keys() and model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]['builtin_options_type'] == 'ObfOptions':
                    model_json['subgraphs'][0]["operators"].append({'builtin_options_type': 'ObfOptions', \
                                                                    "inputs": [model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]["outputs"][0]],
                                                                    "outputs": [out_start_point+i],
                                                                    "ori_inputs": [model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]["ori_inputs"][0]]})
                else:
                    model_json['subgraphs'][0]["operators"].append({'builtin_options_type': 'ObfOptions', \
                                                                    "inputs": [model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]["outputs"][0]],
                                                                    "outputs": [out_start_point+i],
                                                                    "ori_inputs": [model_json['subgraphs'][0]["operators"][rand_extra_pair[0]]["outputs"][0]]})
                end_index = op_i["inputs"].index(model_json['subgraphs'][0]["operators"][rand_extra_start]["outputs"][0])
                op_i["inputs"].remove(model_json['subgraphs'][0]["operators"][rand_extra_start]["outputs"][0])
                op_i["inputs"].insert(end_index, out_start_point+i)


model_path = './tflite_model/'
model_name = opt.model_name + '.tflite'
if opt.model_name == 'gpt2':
    enable_sig = False
else:
    enable_sig = True
interpreter = tf.lite.Interpreter(
 os.path.join(model_path, model_name)
)
interpreter.allocate_tensors()
# --------------------------------------------------
# parse the TFLite model and generate code
# --------------------------------------------------
os.system('flatc -t schema.fbs -- %s' % os.path.join(model_path, model_name))
reduce_size_json(os.path.splitext(model_name)[0] + '.json')
os.system('jsonrepair %s.json --overwrite' % os.path.splitext(model_name)[0])
# for op in interpreter._get_ops_details():
#     print(op)

with open('%s.json' % os.path.splitext(model_name)[0],'r') as f:
    model_json_f = f.read()
model_json = json.loads(model_json_f)

# op_details = interpreter._get_ops_details()
# print(op_details)

# for tensor_details in interpreter.get_tensor_details():
#     print(tensor_details)

ori_model_json = copy.deepcopy(model_json)
ori_model_json = correct_json(interpreter, ori_model_json)

random_shortcut(model_json)
tensor_list = []
for input in interpreter.get_input_details():
    tensor_list.append(input['index'])
for tensor_details in interpreter.get_tensor_details():
    tensor_list.append(tensor_details["index"])
tensor_list.sort()
random_extra(model_json, tensor_list[-1]+10)

model_json = correct_json(interpreter, model_json)

op_count = 0
max_sign = 0  # use it to note how many operators in the model (for assign the "sign" value to the new Op)
for i in range(len(model_json['subgraphs'][0]["operators"])):
    op_count += 1
    model_json['subgraphs'][0]["operators"][i]["sign"] = i
    max_sign = i
    # print(model_json['subgraphs'][0]["operators"][i])
    if model_json['subgraphs'][0]["operators"][i]["builtin_options_type"] in trans_enabled_list:
        model_json['subgraphs'][0]["operators"][i]["trans_enabled"] = 1
    else:
        model_json['subgraphs'][0]["operators"][i]["trans_enabled"] = 0
    if model_json['subgraphs'][0]["operators"][i]["builtin_options_type"] in linear_list:
        model_json['subgraphs'][0]["operators"][i]["linear"] = 1
    else:
        model_json['subgraphs'][0]["operators"][i]["linear"] = 0
    if model_json['subgraphs'][0]["operators"][i]["builtin_options_type"] in nonlinear_enable_list:
        model_json['subgraphs'][0]["operators"][i]["nonlinear_op"] = 1
    else:
        model_json['subgraphs'][0]["operators"][i]["nonlinear_op"] = 0

jsondata = json.dumps(model_json,indent=4,separators=(',', ': '))
file = open('./test.json', 'w')
file.write(jsondata)
file.close()

# ------------------------------------------------
# New code
# ------------------------------------------------

def op_graph(json_data):
    edges_obf = {}
    trans_enabled_labels = {}
    # input_ids = []
    output_ids = []
    linear_enabled_labels = {}
    nonlinear_enabled_labels = {}
    for op_outter in json_data['subgraphs'][0]["operators"]:
        for i in json_data['subgraphs'][0]["outputs"]:
            if i in op_outter["outputs"]:
                output_ids.append(op_outter["sign"])
        if not (op_outter["sign"] in edges_obf):
            edges_obf[op_outter["sign"]] = []
            trans_enabled_labels[op_outter["sign"]] = op_outter["trans_enabled"]
            linear_enabled_labels[op_outter["sign"]] = op_outter["linear"]
            nonlinear_enabled_labels[op_outter["sign"]] = op_outter["nonlinear_op"]
        for out_id in op_outter["outputs"]:
            for op_inner in json_data['subgraphs'][0]["operators"]: 
                if out_id in op_inner["inputs"]:
                    edges_obf[op_outter["sign"]].append(op_inner["sign"])

    return edges_obf, trans_enabled_labels, linear_enabled_labels, nonlinear_enabled_labels

edges_obf, trans_enabled_labels, linear_enabled_labels, nonlinear_enabled_labels = op_graph(model_json)
print("edges_obf", edges_obf)
print("trans_enabled_labels:", trans_enabled_labels)
print("linear_enabled_labels:", linear_enabled_labels)

def find_inj_pair(start_node, max_iter=20):
    level_sign_dict = {}

    first_level_nodes = edges_obf.get(start_node, [])
    # print("first_level_nodes: ", first_level_nodes)
    level_sign_dict[1] = []
    level_sign_dict[2] = []
    second_level_nodes = []
    fused_type = []
    fused_sign_second = False
    try:
        fused_data = get_op_detail_by_sign(model_json, start_node)["builtin_options"]["fused_activation_function"]
    except:
        fused_sign_first = False
        # fused_type.append(None)
    else:
        # print(fused_data)
        fused_sign_first = True
        fused_type.append(fused_data)

    for i in range(len(first_level_nodes)):
        if linear_enabled_labels[first_level_nodes[i]] != 1:
            return None
        if trans_enabled_labels[first_level_nodes[i]] == 1:
            # print("level_sign_dict[1] add: ", first_level_nodes[i])
            level_sign_dict[1].append([first_level_nodes[i]])
        else:
            level_sign_dict[1].append(None)
        if not is_single_edge_destination(edges_obf, first_level_nodes[i]):
            # print(first_level_nodes[i], " is not single-edge destination")
            return None

        if fused_sign_first:
            level_sign_dict[2].append(None)
        else:
            try:
                fused_data = get_op_detail_by_sign(model_json, first_level_nodes[i])["builtin_options"]["fused_activation_function"]
            except:
                fused_sign_second = False
                fused_type.append(None)
            else:
                # print(fused_data)
                fused_sign_second = True
                fused_type.append(fused_data)
            try:
                second_node_tmp_data = edges_obf.get(first_level_nodes[i], [])
            except:
                level_sign_dict[2].append(None)
                continue
            else:
                second_level_nodes.append(second_node_tmp_data)
            level2_tmp = []
            for j in range(len(second_level_nodes[i])):
                if trans_enabled_labels[second_level_nodes[i][j]] == 1:
                    # print("level_sign_dict[1][" + str(j) + "] add: ", second_level_nodes[i][j])
                    level2_tmp.append(second_level_nodes[i][j])
                else:
                    level2_tmp = None
                    break
                if not is_single_edge_destination(edges_obf, second_level_nodes[i][j]):
                    # print(second_level_nodes[i][j], " is not single-edge destination")
                    level2_tmp = None
                    break
                    # level_sign_dict[2].append(first_level_nodes[i])
            level_sign_dict[2].append(level2_tmp)
    paired_node = []
    activation_node = []
    intermediate_nodes = []
    for i in range(len(first_level_nodes)):
        candidate = []
        # print("fused_type: ", fused_type)
        if level_sign_dict[1][i] != None:
            candidate.append(level_sign_dict[1][i])
        if level_sign_dict[2][i] != None:
            # print("the error point: ", level_sign_dict[2][i])
            candidate.append(level_sign_dict[2][i])
        # print("candidate: ", candidate)
        if len(candidate) == 0:
            return None
        elif len(candidate) == 1:
            for j in range(len(candidate[0])):
                # print("fused_type[0]: ", fused_type[0])
                # print("candidate[0][j]: ", candidate[0][j])
                # print("nonlinear_enabled_labels[candidate[0][j]]: ", nonlinear_enabled_labels[candidate[0][j]])
                if fused_type[0] in ['RELU6', 'RELU'] and nonlinear_enabled_labels[candidate[0][j]] == 0:
                    return None
                paired_node.append(candidate[0][j])
                activation_node.append(fused_type[0])
        elif len(candidate) == 2:
            nonlinear_fail_sign = 0
            for j in range(len(candidate[1])):
                if fused_type[i] in ['RELU6', 'RELU'] and nonlinear_enabled_labels[candidate[1][j]] ==0:
                    nonlinear_fail_sign = 1
            if nonlinear_fail_sign == 1:
                random_pick = 0
            else:
                random_pick = random.sample(range(0, 2), 1)[0]
            # print("random_pick: ", random_pick)
            for j in range(len(candidate[random_pick])):
                paired_node.append(candidate[random_pick][j])
                if random_pick == 0:
                    activation_node.append(None)
                else:
                    activation_node.append(fused_type[i])
                    intermediate_nodes.append(candidate[0][i])    # fixed an error in DynaMO
    # print("level_sign_dict: ", level_sign_dict)
    # print("paired_node: ", paired_node)
    # print("activation_node: ", activation_node)
    return paired_node, activation_node, intermediate_nodes


injected_pair = {}
adv_obf_value = {}
activation_type = {}
intermediate_nodes = {}
valid_injected_pair = {}

del_list_non_linear = []
for i in range(op_count*2):
    start_node = pick_random_node_with_label_one(trans_enabled_labels)
    # print("start_node: ", start_node)
    # non_linear_op_sign = 1
    if find_inj_pair(start_node) != None:
        paired_node, activation_node, intermediate = find_inj_pair(start_node)
        # for j in range(len(paired_node)):
        #     print(j)
        #     print(activation_type)
        #     print(paired_node)
        #     if activation_type[j] in ['RELU6', 'RELU'] and paired_node[j] not in nonlinear_op_list:
        #         non_linear_op_sign=0
        # if non_linear_op_sign == 1:
        injected_pair[start_node] = paired_node
        activation_type[start_node] = activation_node
        intermediate_nodes[start_node] = intermediate
        # obf_value = 0.8
        # obf_value = random.uniform(0.8, 0.9)
        # obf_value = 0.5
        obf_value = 1.0 / opt.hash_deobf_value
        adv_obf_value[start_node] = obf_value

        # ------------------------------------------------------------------------------
        # # we need to add the non-linear op to the end_node, but this end_node cannot be
        # # the start_node of injected_pair. So we need to delete invalid pairs
        # ------------------------------------------------------------------------------
        for m in range(len(activation_node)):
            # print("value[i]: ", value[i])
            if activation_node[m] in ['RELU6']:
                # print("injected_pair[key][i]: ", injected_pair[key][i])
                del_list_non_linear.append(injected_pair[start_node][m])

        if len(del_list_non_linear) != 0:
            for n in range(len(del_list_non_linear)):
                valid_injected_pair = {key:val for key, val in injected_pair.items() if key != del_list_non_linear[n]}
        else:
            valid_injected_pair = injected_pair
        # print(len(valid_injected_pair))
        if len(valid_injected_pair) >= opt.verify_nb:
            break

if len(valid_injected_pair) < opt.verify_nb:
    print("-----------------------------------")
    raise ValueError("Cannot find enough valid pairs")

for i in range(len(del_list_non_linear)):
    injected_pair = {key:val for key, val in injected_pair.items() if key != del_list_non_linear[i]}
    activation_type = {key:val for key, val in activation_type.items() if key != del_list_non_linear[i]}
    adv_obf_value = {key:val for key, val in adv_obf_value.items() if key != del_list_non_linear[i]}
    intermediate_nodes = {key:val for key, val in intermediate_nodes.items() if key != del_list_non_linear[i]}

print("should delete: ", del_list_non_linear)
print("injected_pair: ", injected_pair)
print("activation_type: ", activation_type)
print("adv_obf_value: ", adv_obf_value)
print("intermediate_nodes: ", intermediate_nodes)


# ------------------------------------------------
# remove unneeded ObfOptions (not used to verify the signature), for efficiency
# ------------------------------------------------

injected_pair_values_list = []
for value in injected_pair.values():
    if isinstance(value, list):
        injected_pair_values_list.extend(value)
    else:
        injected_pair_values_list.append(value)

intermediate_nodes_values_list = []
for value in intermediate_nodes.values():
    if isinstance(value, list):
        intermediate_nodes_values_list.extend(value)
    else:
        intermediate_nodes_values_list.append(value)

del_list = []
for i in range(len(model_json['subgraphs'][0]["operators"])):
    op_i = model_json['subgraphs'][0]["operators"][i]
    if op_i["builtin_options_type"] == "ObfOptions":
        if op_i["sign"] not in injected_pair_values_list and \
            op_i["sign"] not in intermediate_nodes_values_list and \
            op_i["sign"] not in list(injected_pair.keys()) and \
            len(op_i["inputs"]) == 1 and len(op_i["outputs"]) == 1:

            for j in range(len(model_json['subgraphs'][0]["operators"])):
                op_j = model_json['subgraphs'][0]["operators"][j]
                if op_i["outputs"][0] in op_j["inputs"]:
                    end_index = op_j["inputs"].index(op_i["outputs"][0])
                    op_j["inputs"].remove(op_i["outputs"][0])
                    op_j["inputs"].insert(end_index, op_i["inputs"][0])
                    print("delete the ObfOptions: ", op_i["sign"])
                    print("delete the input id: ", op_i["outputs"][0], " from ", op_j["inputs"])
                    print("add the input id: ", op_i["inputs"][0])
            del_list.append(model_json['subgraphs'][0]["operators"][i])


for del_list_i in del_list:
    model_json['subgraphs'][0]["operators"].remove(del_list_i)

# ------------------------------------------------
# End New code
# ------------------------------------------------

inout_list = []
for i in range(len(model_json['subgraphs'][0]["operators"])):
    # print(model_json['subgraphs'][0]["operators"][i])
    for j in range(len(model_json['subgraphs'][0]["operators"][i]['outputs'])):
        inout_list.append(model_json['subgraphs'][0]["operators"][i]['outputs'][j])

for input in interpreter.get_input_details():
    inout_list.append(input['index'])

# for output in interpreter.get_output_details():
#     inout_list.append(output['index'])

# -----------------------------------------------------------------------------------
# Some FC layers (withpout_filter_FC) use previous outputs as the filter (like GPT2),
# Should add ObfOptions to perform the linear transformation on the filter
# -----------------------------------------------------------------------------------
nb_adding_obf = 0 # use to count how many additional ObfOptions added, and assign the output_id
for key, value in injected_pair.items():
    op_i = get_op_detail_by_sign(model_json, key)
    # print(op_i["builtin_options_type"])
    # print(len(op_i["inputs"]))
    if op_i["builtin_options_type"] == "FullyConnectedOptions" and op_i["inputs"][1] in inout_list:
        print("select a FC as the start obfuscation node but don't has the filter")
        prev_node_is_obf = False
        # --------------------------------------------------------------------------
        # If the start of the withpout_filter_FC is connect to the end of Obf node,
        # direclty modify the fc_filter_obf value to apply the filter transformation
        # --------------------------------------------------------------------------
        for op_j in model_json['subgraphs'][0]["operators"]:
            if op_i["inputs"][1] in op_j["outputs"] and op_j["builtin_options_type"] == "ObfOptions":
                print("Previous node is obf")
                if "fc_filter_obf" in op_j.keys():
                    op_j["fc_filter_obf"] = op_j["fc_filter_obf"] * adv_obf_value[key]
                else:
                    op_j["fc_filter_obf"] = adv_obf_value[key]
                # max_sign += 1
                # model_json['subgraphs'][0]["operators"].append({'builtin_options_type': 'ObfOptions', \
                #                                 "inputs": [op_i["inputs"][1]],
                #                                 "outputs": [tensor_list[-1]+10+opt.extra_layer],
                #                                 "ori_inputs": [op_j["ori_inputs"][0]],
                #                                 "fc_filter_obf": adv_obf_value[key],
                #                                 "sign": max_sign})
                prev_node_is_obf = True
                break
        if not prev_node_is_obf:
            print("Previous node is not obf")
            max_sign += 1
            nb_adding_obf += 1
            model_json['subgraphs'][0]["operators"].append({'builtin_options_type': 'ObfOptions', \
                                                            "inputs": [op_i["inputs"][1]],
                                                            "outputs": [tensor_list[-1]+10+opt.extra_layer+nb_adding_obf],
                                                            "ori_inputs": [op_i["inputs"][1]],
                                                            "fc_filter_obf": adv_obf_value[key],
                                                            "sign": max_sign})
            inout_list.append(tensor_list[-1]+10+opt.extra_layer+nb_adding_obf)
            print("Add a ObfOptions to the filter position of the FC layer, input is %d, output is %d" % (op_i["inputs"][1], tensor_list[-1]+10+opt.extra_layer+nb_adding_obf))
            # end_index = op_i["inputs"].index(model_json['subgraphs'][0]["operators"][rand_extra_start]["outputs"][0])
            op_i["inputs"].remove(op_i["inputs"][1])
            op_i["inputs"].insert(1, tensor_list[-1]+10+opt.extra_layer+nb_adding_obf)



jsontext, unknown_config = lib_generator(model_json, ori_model_json, interpreter, inout_list, injected_pair, adv_obf_value, activation_type, intermediate_nodes, opt.hash_deobf_value)

# --------------------------------------------------
# dinamic config & build
# --------------------------------------------------
currentPath = os.getcwd().replace('\\','/')
# os.chdir('./tensorflow-2.9.1/')
# os.system("bash build.sh")
os.chdir(currentPath)
# print(inout_list)
for op in jsontext['oplist']:
    del_list = []
    # print("input:", op['input'])
    for i in range(len(op['input'])):
        if not (op['input'][i] in inout_list):
            # print("not in inout_list:", op['input'][i])
            del_list.append(op['input'][i])
    # print("del:", del_list)
    for j in range(len(del_list)):
        op['input'].remove(del_list[j])

    out_node = op['output'][0]
    try:
        model_json['subgraphs'][0]["tensors"][out_node]["type"]
    except:
        op['type'] = "FLOAT32"
    else:
        op['type'] = model_json['subgraphs'][0]["tensors"][out_node]["type"]
    try:
        model_json['subgraphs'][0]["tensors"][out_node]["quantization"]
    except:
        op["quantization"] = {}
    else:
        op["quantization"] = model_json['subgraphs'][0]["tensors"][out_node]["quantization"]

input_list = model_json['subgraphs'][0]['inputs']
jsontext['inputs'] = []
for i in range(len(input_list)):
    try:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    except:
        tensor_type = "FLOAT32"
    else:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    jsontext['inputs'].append({'name': 'serving_default_x:'+str(i), 'type': tensor_type, 'quantization': model_json['subgraphs'][0]["tensors"][input_list[i]]["quantization"]})

output_list = model_json['subgraphs'][0]['outputs']
jsontext['outputs'] = []
for i in range(len(output_list)):
    try:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    except:
        tensor_type = "FLOAT32"
    else:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    jsontext['outputs'].append({'name': 'PartitionedCall:'+str(i), 'type': tensor_type, 'quantization': model_json['subgraphs'][0]["tensors"][output_list[i]]["quantization"]})

jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))

file = open('./obfjson/model' + '_' + opt.model_name + '.json', 'w')
file.write(jsondata)
file.close()

model_assembler(interpreter, './obfjson/model' + '_' + opt.model_name + '.json', opt.free_unused_data, enable_sig=enable_sig, executable=opt.executable)
dinamic_config(unknown_config, './obfjson/model' + '_' + opt.model_name + '.json', os.path.join(model_path, model_name), enable_sig=enable_sig, executable=opt.executable)
# os.system('python eval.py --model_name=' + opt.model_name + ' --latency=True')
