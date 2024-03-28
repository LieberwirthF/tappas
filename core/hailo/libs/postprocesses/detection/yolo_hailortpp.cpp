#include <regex>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"
#include "json_config.hpp"
#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "common/labels/coco_eighty.hpp"

static const std::string DEFAULT_YOLOV5S_OUTPUT_LAYER = "yolov5s_nv12/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5m_wo_spp_60p/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER = "yolov5m_vehicles/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV7TINY_OUTPUT_LAYER = "yolov7_tiny/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV8S_OUTPUT_LAYER = "yolov8s/yolov8_nms_postprocess";
static const std::string DEFAULT_YOLOV8M_OUTPUT_LAYER = "yolov8m/yolov8_nms_postprocess";

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

YoloParamsNMS *init(const std::string config_path, const std::string function_name)
{
    YoloParamsNMS *params;
    if (!fs::exists(config_path))
    {
        params = new YoloParamsNMS(common::coco_eighty);
        return params;
    }
    else
    {
        params = new YoloParamsNMS();
        char config_buffer[4096];
        const char *json_schema = R""""({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "detection_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "max_boxes": {
            "type": "integer"
            },
            "labels": {
            "type": "array",
            "items": {
                "type": "string"
                }
            },
            "output_layer_name": {
            "type": "string"
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
            if (doc_config_json.HasMember("detection_threshold")) {
                params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            }
            if (doc_config_json.HasMember("max_boxes")) {
                params->max_boxes = doc_config_json["max_boxes"].GetInt();
                params->filter_by_score = true;
            }
            if (doc_config_json.HasMember("output_layer_name")) {
                params->output_layer_name = doc_config_json["output_layer_name"].GetString();
            }
        }
        fclose(fp);
    }
    return params;
}

void free_resources(void *params_void_ptr)
{
    YoloParamsNMS *params = reinterpret_cast<YoloParamsNMS *>(params_void_ptr);
    delete params;
}

static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};

void yolov5(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5s_nv12(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov8s(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8S_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov8m(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV8M_OUTPUT_LAYER), common::coco_eighty);
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
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_VEHICLES_OUTPUT_LAYER), yolo_vehicles_labels);
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

void yolov7tiny(HailoROIPtr roi, void *params_void_ptr)
{
    if (!roi->has_tensors())
    {
        return;
    }
    YoloParamsNMS *params = reinterpret_cast<YoloParamsNMS *>(params_void_ptr);
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    // find the nms tensor
    for (auto tensor : tensors)
    {
        if (std::regex_search(tensor->name(), std::regex("nms_postprocess"))) 
        {
            std::string output_layer_name;
            if (params->output_layer_name.empty()){
                output_layer_name = DEFAULT_YOLOV7TINY_OUTPUT_LAYER;
            } else {
                output_layer_name = params->output_layer_name;
            }

            auto post = HailoNMSDecode(roi->get_tensor(output_layer_name), params->labels, params->detection_threshold, params->max_boxes, params->filter_by_score);
            auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
            hailo_common::add_detections(roi, detections);
        }
    }
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    yolov7tiny(roi, params_void_ptr);
}
