/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
*
* created by Fridtjof Lieberwirth (lieberwirth@dresearch-fe.de)
**/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS

class CenterNetParams
{
public:
    float detection_threshold;
    float iou_threshold;
    std::map<std::uint8_t, std::string> labels;
    uint num_classes;
    uint max_boxes;
    std::string hm_layer_name;
    std::string wh_layer_name;
    std::string reg_layer_name;
};

CenterNetParams *init(std::string config_path, std::string func_name);
void free_resources(void *params_void_ptr);

void filter(HailoROIPtr roi, void *params_void_ptr);
void centernet(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS