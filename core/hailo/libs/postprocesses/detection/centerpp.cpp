
/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
*
* created by Fridtjof Lieberwirth (lieberwirth@dresearch-fe.de)
**/

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "centerpp.hpp"
#include "hailo_xtensor.hpp"
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "common/nms.hpp"
#include "json_config.hpp"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcontainer.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xshape.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

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

/**
 * @brief get top k centers
 *
 * @param scores output tensors of scores
 * @param k take k best scores and ignore the others
 * @return std::pair<xt::xarray<int>, xt::xarray<uint8_t>> pair of indices of scores and scores
 */
std::pair<xt::xarray<int>, xt::xarray<uint8_t>> top_k_centers(HailoTensorPtr scores, const int k)
{
    // Adapt the tensor into an xarray of proper shape and size
    auto xscores = common::get_xtensor(scores);

    // Get the indices of the top k scoring cells
    int size = scores->size();
    xt::xarray<uint8_t> xscores_view = xt::reshape_view(xscores, {1, size});
    xt::xarray<int> topk_score_indices = common::top_k(xscores_view, k);
    topk_score_indices = xt::flatten(topk_score_indices);

    // We want a flattened view of the scores to sort by
    auto flat_scores = xt::flatten(xscores);

    // Using the top k indices, get the top k scores
    auto topk_scores = xt::view(flat_scores, xt::keep(topk_score_indices));

    // Return the top scores and their indices
    return std::pair<xt::xarray<int>, xt::xarray<uint8_t>>(std::move(topk_score_indices), std::move(topk_scores));
}

/**
 * @brief Extract features from tensor
 *
 * @param tensor output tensor
 * @param indices indices to keep
 * @return xt::xarray<uint8_t> features
 */
xt::xarray<uint8_t> gather_features_from_tensor(HailoTensorPtr tensor, xt::xarray<int> &indices)
{
    // Adapt the tensor into an xarray of proper shape and size:
    auto xtensor = common::get_xtensor(tensor);

    // Extract the top k keypoints using the given indices:
    // Use a reshaped view of the given tensor so that features can be gathered
    auto transposed_tensor = xt::reshape_view(xtensor, {tensor->width() * tensor->height(), tensor->features()});

    // Gather the features using the given indices
    auto features = xt::view(transposed_tensor, xt::keep(indices), xt::all());

    return features;
}

/**
 * @brief Box and keypoint extraction
 *
 * @param scores array of scores
 * @param center_offsets array of center offsets
 * @param center_wh centernet boxes width height
 * @param cell_x_indices array of indices x of the cells
 * @param cell_y_indices array of indices y of the cells
 * @param score_threshold threshold for score filtering
 * @param image_size image width/height (the netowork input tensor is square)
 * @return xt::xarray<float>
 */
xt::xarray<float> build_boxes_centernet(xt::xarray<float> &scores,
                                         xt::xarray<float> &center_offsets,
                                         xt::xarray<float> &center_wh,
                                         xt::xarray<int> &cell_x_indices,
                                         xt::xarray<int> &cell_y_indices,
                                         const float score_threshold,
                                         const int image_size)
{
    // Here we need to calculate the min and max of the box. The cell index + offset gives the real
    // center of the box, then subtracting half of the width/height will get the xmin/ymin.
    xt::xarray<float> xmin = (cell_x_indices + xt::col(center_offsets, 0) - (xt::col(center_wh, 0) * 0.5));
    xt::xarray<float> ymin = (cell_y_indices + xt::col(center_offsets, 1) - (xt::col(center_wh, 1) * 0.5));
    xt::xarray<float> xmax = xmin + xt::col(center_wh, 0);
    xt::xarray<float> ymax = ymin + xt::col(center_wh, 1);

    // Stack the box parameters along axis 1, ending with an array of shape { k, 4 }
    xt::xarray<float> detection_boxes = xt::stack(xt::xtuple(xmin, ymin, xmax, ymax), 1);
    return detection_boxes;
}

/**
 * @brief Detection/landmarks encoding
 *
 * @param objects vector of the detected instances
 * @param scores scores array
 * @param detection_boxes boxes detected
 * @param center_wh centernet boxes width height
 * @param keypoints keypoints
 * @param score_threshold threshold for score filtering
 * @param max_detections max number of best results
 * @param labels map with label_id and label name
 * @param image_size image width/height (the netowork input tensor is square)
 */
void encode_boxes_centernet(std::vector<HailoDetection> &objects,
                            xt::xarray<float> &scores,
                            xt::xarray<float> &detection_boxes,
                            xt::xarray<float> &center_wh,
                            const float score_threshold,
                            const int max_detections,
                            std::map<std::uint8_t, std::string> labels,
                            const int image_size)
{
    // The detection meta will hold the following items:
    float confidence, w, h, xmin, ymin = 0.0f;
    // label_id hardcoded since it's not clear where the labl_id can be extracted from the tensor
    std::uint8_t label_id = 0;
    xt::xarray<float> wh_scaled = center_wh / image_size;
    // Iterate over our top k results
    for (int index = 0; index < max_detections; index++)
    {
        confidence = scores(index);
        // If the confidence is below our threshold, then skip it
        if (confidence < score_threshold)
            continue;

        w = wh_scaled(index, 0); // Box width, relative to image size
        h = wh_scaled(index, 1); // Box height, relative to image size
        // The xmin and ymin we can take form the detection_boxes
        xmin = detection_boxes(index, 0) / image_size;
        ymin = detection_boxes(index, 1) / image_size;

        // Once all parameters are calculated, push them into the meta
        // Class = 1 since centernet only detects people
        HailoDetection detection(HailoBBox(xmin, ymin, w, h), label_id, labels[label_id], confidence);

        objects.emplace_back(std::move(detection)); // Push the detection to the objects vector
    }
}

/**
 * @brief centernet post process
 *
 * @param roi region of interest
 * @param params CenterNetParams
 * @return std::vector<HailoDetection> the detected objects
 */
std::vector<HailoDetection> centernet_postprocess(HailoROIPtr roi,
                                                   CenterNetParams *params)
{
    std::vector<HailoDetection> objects; // The detection meta we will eventually return
    // Extract the 6 output tensors:
    // Center heatmap tensor with scaling and offset tensors for person detection
    HailoTensorPtr center_heatmap = roi->get_tensor(params->hm_layer_name);
    HailoTensorPtr center_width_height = roi->get_tensor(params->wh_layer_name);
    HailoTensorPtr center_offset = roi->get_tensor(params->reg_layer_name);

    // detection box encoding
    // From the center_heatmap tensor, we want to extract the top k centers with the highest score
    auto top_scores = top_k_centers(center_heatmap, params->max_boxes);                                  // Returns both the top scores and their indices
    xt::xarray<int> topk_score_indices = top_scores.first;                               // Separate out the top score indices
    xt::xarray<uint8_t> topk_scores = top_scores.second;                                 // Separate out the top scores
    xt::xarray<int> topk_scores_y_index = topk_score_indices / center_heatmap->height(); // Find the y index of the cells
    xt::xarray<int> topk_scores_x_index = topk_score_indices % center_heatmap->width();  // Find the x index of the cells

    // With the top k indices in hand, we can now extract the corresponding center offsets and widths/heights
    auto topk_center_offset = gather_features_from_tensor(center_offset, topk_score_indices);   // Use the top k indices from earlier
    auto topk_center_wh = gather_features_from_tensor(center_width_height, topk_score_indices); // Use the top k indices from earlier

    // Now that we have our top k features, we can rescale them to dequantize
    xt::xarray<float> topk_scores_rescaled = common::dequantize(topk_scores,
                                                                center_heatmap->vstream_info().quant_info.qp_scale, center_heatmap->vstream_info().quant_info.qp_zp);
    xt::xarray<float> topk_center_offset_rescaled = common::dequantize(topk_center_offset,
                                                                       center_offset->vstream_info().quant_info.qp_scale, center_offset->vstream_info().quant_info.qp_zp);

    xt::xarray<float> topk_center_wh_rescaled = common::dequantize(topk_center_wh,
                                                                   center_width_height->vstream_info().quant_info.qp_scale, center_width_height->vstream_info().quant_info.qp_zp);

    const int image_size = center_heatmap->width(); // We want the boxes to be of relative size to the original image
    // Build up the detection boxes
    auto bboxes = build_boxes_centernet(topk_scores_rescaled,
                                        topk_center_offset_rescaled,
                                        topk_center_wh_rescaled,
                                        topk_scores_x_index, topk_scores_y_index, params->detection_threshold, image_size);

    //-------------------------------
    // RESULTS ENCODING
    //-------------------------------

    // Encode the individual boxes/keypoints and package them into the meta
    encode_boxes_centernet(objects,
                           topk_scores_rescaled,
                           bboxes,
                           topk_center_wh_rescaled,
                           params->detection_threshold,
                           params->max_boxes,
                           params->labels,
                           image_size);

    // Perform nms to throw out similar detections
    common::nms(objects, params->iou_threshold);

    return objects;
}

/**
 * @brief Perform post process and add the detected objects to the roi object
 *
 * @param roi region of interest
 */
void centernet(HailoROIPtr roi, void *params_void_ptr)
{
    if (roi->has_tensors())
    {
        CenterNetParams *params = reinterpret_cast<CenterNetParams *>(params_void_ptr);

        auto detections = centernet_postprocess(roi, params);

        // Update the roi with the found detections.
        hailo_common::add_detections(roi, detections);
    }
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    centernet(roi, params_void_ptr);
}

CenterNetParams *init(const std::string config_path, const std::string function_name)
{
    CenterNetParams *params;
    if (!fs::exists(config_path))
    {
        std::cerr << "Please pass a config." << std::endl;
        return params;
    }
    else
    {
        params = new CenterNetParams;
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
            "max_boxes": {
            "type": "integer"
            },
            "hm_layer_name": {
            "type": "string"
            },
            "wh_layer_name": {
            "type": "string"
            },
            "reg_layer_name": {
            "type": "string"
            },
            "labels": {
            "type": "array",
            "items": {
                "type": "string"
                }
            }
        },
        "required": [
            "iou_threshold",
            "detection_threshold",
            "max_boxes",
            "hm_layer_name",
            "wh_layer_name",
            "reg_layer_name",
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
            params->iou_threshold = doc_config_json["iou_threshold"].GetFloat();
            params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            params->hm_layer_name = doc_config_json["hm_layer_name"].GetString();
            params->wh_layer_name = doc_config_json["wh_layer_name"].GetString();
            params->reg_layer_name = doc_config_json["reg_layer_name"].GetString();
            params->max_boxes = doc_config_json["max_boxes"].GetInt();
        }
        fclose(fp);
    }
    return params;
}

void free_resources(void *params_void_ptr)
{
    CenterNetParams *params = reinterpret_cast<CenterNetParams *>(params_void_ptr);
    delete params;
}
