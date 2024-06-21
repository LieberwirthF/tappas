#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "common/labels/coco_eighty.hpp"
#include "json_config.hpp"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

static const std::string DEFAULT_YOLOV5_OUTPUT_LAYER = "yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5m_wo_spp_60p/yolov5_nms_postprocess";
static const std::string DEFAULT_YOLOV8S_OUTPUT_LAYER = "yolov8s/yolov8_nms_postprocess";
static const std::string DEFAULT_YOLOV8M_OUTPUT_LAYER = "yolov8m/yolov8_nms_postprocess";

static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};

void yolov7(HailoROIPtr roi, void *params_void_ptr)
{
    YoloV7Params *params = reinterpret_cast<YoloV7Params *>(params_void_ptr);

    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), params->labels);

    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m(HailoROIPtr roi)
{
    if (!roi->has_tensors())
    {
        return;
    }
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
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

void filter(HailoROIPtr roi)
{
    yolov5(roi);
}

YoloV7Params *init(const std::string config_path, const std::string function_name)
{
    YoloV7Params *params;
    if (!fs::exists(config_path))
    {
        std::ostringstream oss;
        oss << "No config found" << std::endl;

        throw std::runtime_error(oss.str());
        return params;
    }
    else
    {
        params = new YoloV7Params;
        char config_buffer[4096];
        const char *json_schema = R""""({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "iou_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "detection_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "output_activation": {
            "type": "string"
            },
            "label_offset": {
            "type": "integer"
            },
            "max_boxes": {
            "type": "integer"
            },
            "anchors": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                "type": "integer"
                }
            }
            },
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
            // parse anchors
            // auto config_anchors = doc_config_json["anchors"].GetArray();
            // std::vector<std::vector<int>> anchors_vec;
            // for (uint j = 0; j < config_anchors.Size(); j++)
            // {
            //     uint size = config_anchors[j].GetArray().Size();
            //     std::vector<int> anchor;
            //     for (uint k = 0; k < size; k++)
            //     {
            //         anchor.push_back(config_anchors[j].GetArray()[k].GetInt());
            //     }
            //     anchors_vec.push_back(anchor);
            // }

            // params->anchors_vec = anchors_vec;
            // // set the params
            // params->iou_threshold = doc_config_json["iou_threshold"].GetFloat();
            // params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            // params->output_activation = doc_config_json["output_activation"].GetString();
            // params->label_offset = doc_config_json["label_offset"].GetInt();
            // params->max_boxes = doc_config_json["max_boxes"].GetInt();
            // if (params->output_activation != "sigmoid" && params->output_activation != "none")
            // {
            //     std::ostringstream oss;
            //     oss << "config output activation do not match! output activation: "
            //         << params->output_activation << std::endl;
            //     throw std::runtime_error(oss.str());
            // }
        }
        fclose(fp);
    }
    return params;
}
// void YoloV7Params::check_params_logic(uint num_classes_tensors)
// {
//     if (labels.size() - 1 != num_classes_tensors)
//     {
//         std::ostringstream oss;
//         oss << "config class labels do not match output tensors! config labels size: "
//             << labels.size() - 1 << " tensors num classes: " << num_classes_tensors << std::endl;
//         throw std::runtime_error(oss.str());
//     }
// }

void free_resources(void *params_void_ptr)
{
    YoloV7Params *params = reinterpret_cast<YoloV7Params *>(params_void_ptr);
    delete params;
}
