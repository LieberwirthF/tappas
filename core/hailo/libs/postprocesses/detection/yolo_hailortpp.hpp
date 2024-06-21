/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"


__BEGIN_DECLS

class YoloV7Params
{
public:
    // float iou_threshold;
    // float detection_threshold;
    std::map<std::uint8_t, std::string> labels;
    // uint num_classes;
    // uint max_boxes;
    // std::vector<std::vector<int>> anchors_vec;
    // std::string output_activation; // can be "none" or "sigmoid"
    // int label_offset;
    // YoloV7Params() : iou_threshold(0.45f), detection_threshold(0.3f), output_activation("none"), label_offset(1) {}
    // void check_params_logic(uint num_classes_tensors);
};

YoloV7Params *init(std::string config_path, std::string func_name);
void free_resources(void *params_void_ptr);

void yolov7(HailoROIPtr roi, void *params_void_ptr);
void filter(HailoROIPtr roi);
void yolov5(HailoROIPtr roi);
void yolov5m(HailoROIPtr roi);
void yolov8s(HailoROIPtr roi);
void yolov8m(HailoROIPtr roi);
void yolox(HailoROIPtr roi);
void yolov5_no_persons(HailoROIPtr roi);
void yolov5m_vehicles(HailoROIPtr roi);
__END_DECLS
