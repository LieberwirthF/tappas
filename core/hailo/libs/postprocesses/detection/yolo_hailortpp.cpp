#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "common/labels/coco_eighty.hpp"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5_nms_postprocess";

static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};

void yolov5(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);

    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), params->labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolox(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor("yolox_nms_postprocess"), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m_vehicles(HailoROIPtr roi)
{
    // auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), yolo_vehicles_labels, 0.4, DEFAULT_MAX_BOXES, true);
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), yolo_vehicles_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5_no_persons(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    for (auto it = detections.begin(); it != detections.end();)
    {
        if (it->get_label() == "person")
        {
            it = detections.erase(it);
        }
        else
        {
            ++it;
        }
    }
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi, oid *params_void_ptr)
{
    yolov5(roi, params_void_ptr);
}

YoloParams *init(const std::string config_path, const std::string function_name)
{
    YoloParams *params;
    if (!fs::exists(config_path))
    {
        std::cerr << "Please pass a config." << std::endl;
        return params;
    }
    else
    {
        params = new YoloParams;
        char config_buffer[4096];
        const char *json_schema = R""""({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "labels"
        ]
        })"""";

        std::FILE *fp = fopen(config_path.c_str(), "r");
        if (fp == nullptr)
        {
            throw std::runtime_error("JSON config file is not valid");
        }
        rapidjson::FileReadStream stream(fp, config_buffer, sizeof(config_buffer));
        bool valid = common::validate_json_with_schema(stream, json_schema);
        if (valid)
        {
            rapidjson::Document doc_config_json;
            doc_config_json.ParseStream(stream);

            // parse labels
            auto labels = doc_config_json["labels"].GetArray();
            uint i = 0;
            for (auto &v : labels)
            {
                params->labels.insert(std::pair<std::uint8_t, std::string>(i, v.GetString()));
                i++;
            }
            // set the params
            // params->iou_threshold = doc_config_json["iou_threshold"].GetFloat();
            // params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            // params->hm_layer_name = doc_config_json["hm_layer_name"].GetString();
            // params->wh_layer_name = doc_config_json["wh_layer_name"].GetString();
            // params->reg_layer_name = doc_config_json["reg_layer_name"].GetString();
            // params->max_boxes = doc_config_json["max_boxes"].GetInt();
        }
        fclose(fp);
    }
    return params;
}

void free_resources(void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    delete params;
}