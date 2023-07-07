#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "net.h"
#include "benchmark.h"
#include "ros/ros.h"
#include "fastestdet.h"
#include "std_msgs/String.h"
#include <sstream>

// #define PARAM_PATH "/home/hzq/ncnn/build/tools/onnx/fastest_light.param"
// #define BIN_PATH "/home/hzq/ncnn/build/tools/onnx/fastest_light.bin"

#define PARAM_PATH "/home/hzq/ncnn-ros/src/ncnn-ros/model_zoo/fastestdet/FastestDet.param"
#define BIN_PATH "/home/hzq/ncnn-ros/src/ncnn-ros/model_zoo/fastestdet/FastestDet.bin"

void pretty_print(const ncnn::Mat &m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

float Sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float Tanh(float x)
{
    return 2.0f / (1.0f + exp(-2 * x)) - 1;
}

class TargetBox
{
private:
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };

public:
    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;

    float area() { return GetWidth() * GetHeight(); };
};

float IntersectionArea(const TargetBox &a, const TargetBox &b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

bool scoreSort(TargetBox a, TargetBox b)
{
    return (a.score > b.score);
}

// NMS处理
int nmsHandle(std::vector<TargetBox> &src_boxes, std::vector<TargetBox> &dst_boxes)
{
    std::vector<int> picked;

    sort(src_boxes.begin(), src_boxes.end(), scoreSort);

    for (int i = 0; i < src_boxes.size(); i++)
    {
        int keep = 1;
        for (int j = 0; j < picked.size(); j++)
        {
            // 交集
            float inter_area = IntersectionArea(src_boxes[i], src_boxes[picked[j]]);
            // 并集
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > 0.45 && src_boxes[i].category == src_boxes[picked[j]].category)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    for (int i = 0; i < picked.size(); i++)
    {
        dst_boxes.push_back(src_boxes[picked[i]]);
    }

    return 0;
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "");
    if (argc != 2)
    {
        fprintf(stderr, "no image input\n");
        return -1;
    }
    ros::init(argc, argv, "detector_fastestdet");
    ros::NodeHandle node_fastest;

    ros::Publisher pub = node_fastest.advertise<std_msgs::String>("out", 10);

    std_msgs::String msg;
    ros::Rate r(1);

    // 阈值
    float thresh = 0.65;

    // 模型输入宽高
    int input_width = 352;
    int input_height = 352;

    // 加载模型
    ncnn::Net net;
    net.load_param(PARAM_PATH);
    net.load_model(BIN_PATH);
    printf("ncnn model load sucess...\n");

    // 加载图片
    double start = ncnn::get_current_time();
    const char *images = argv[1];
    cv::Mat img = cv::imread(images);
    int img_width = img.cols;
    int img_height = img.rows;

    // resize of input image data
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR,
                                                    img.cols, img.rows, input_width, input_height);
    // pretty_print(input);
    // Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    // creat extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(1);

    // set input tensor
    ex.input("input.1", input);

    // get output tensor
    ncnn::Mat output;
    ex.extract("758", output);
    printf("output: %d, %d, %d\n", output.c, output.h, output.w);

    // handle output tensor
    std::vector<TargetBox> target_boxes;

    for (int h = 0; h < output.h; h++)
    {
        for (int w = 0; w < output.h; w++)
        {
            // 前景概率
            int obj_score_index = (0 * output.h * output.w) + (h * output.w) + w;
            float obj_score = output[obj_score_index];

            // 解析类别
            int category;
            float max_score = 0.0f;
            for (size_t i = 0; i < class_num; i++)
            {
                int obj_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w;
                float cls_score = output[obj_score_index];
                if (cls_score > max_score)
                {
                    max_score = cls_score;
                    category = i;
                }
            }
            float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

            // 阈值筛选
            if (score > thresh)
            {
                // 解析坐标
                int x_offset_index = (1 * output.h * output.w) + (h * output.w) + w;
                int y_offset_index = (2 * output.h * output.w) + (h * output.w) + w;
                int box_width_index = (3 * output.h * output.w) + (h * output.w) + w;
                int box_height_index = (4 * output.h * output.w) + (h * output.w) + w;

                float x_offset = Tanh(output[x_offset_index]);
                float y_offset = Tanh(output[y_offset_index]);
                float box_width = Sigmoid(output[box_width_index]);
                float box_height = Sigmoid(output[box_height_index]);

                float cx = (w + x_offset) / output.w;
                float cy = (h + y_offset) / output.h;

                int x1 = (int)((cx - box_width * 0.5) * img_width);
                int y1 = (int)((cy - box_height * 0.5) * img_height);
                int x2 = (int)((cx + box_width * 0.5) * img_width);
                int y2 = (int)((cy + box_height * 0.5) * img_height);

                target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
            }
        }
    }

    // NMS处理
    std::vector<TargetBox> nms_boxes;
    nmsHandle(target_boxes, nms_boxes);

    // 打印耗时

    // draw result
    for (size_t i = 0; i < nms_boxes.size(); i++)
    {
        TargetBox box = nms_boxes[i];
        printf("x1:%d y1:%d x2:%d y2:%d  %s:%.2f%%\n", box.x1, box.y1, box.x2, box.y2, class_names[box.category], box.score * 100);
        char buffer[100];
        sprintf(buffer, "x1:%d y1:%d x2:%d y2:%d  %s:%.2f%%\n", box.x1, box.y1, box.x2, box.y2, class_names[box.category], box.score * 100);
        msg.data = buffer;
        pub.publish(msg);
        // ROS_INFO(msg.data);

        cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 0, 255), 2);
        cv::putText(img, class_names[box.category], cv::Point(box.x1, box.y1), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
    }
    double end = ncnn::get_current_time();
    double time = end - start;
    printf("Time:%7.2f ms\n", time);

    cv::imwrite("result.jpg", img);

    return 0;
}
