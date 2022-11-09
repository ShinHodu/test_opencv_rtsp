#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "opencv2/opencv_modules.hpp"

#include <stdio.h>
#include <vector>
#include <numeric>
#include <time.h>
#include <Windows.h>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/core/cuda/warp.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>



using namespace std;
using namespace cv;

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 5;

// colors for bounding boxes
const cv::Scalar colors[] = {
    { 0, 255, 255 },
    { 255, 255, 0 },
    { 0, 255, 0 },
    { 255, 0, 0 }
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);


void gpuVideoThread(int threadNum, int channel4gpu, string videoPath, int bimshow)
{
    if (threadNum < channel4gpu)
    {
        cv::cuda::setDevice(0);
    }
    else if (threadNum < channel4gpu * 2)
    {
        cv::cuda::setDevice(1);
    }
    else if (threadNum < channel4gpu * 3)
    {
        cv::cuda::setDevice(2);
    }
    else if (threadNum < channel4gpu * 4)
    {
        cv::cuda::setDevice(3);
    }

    // 비디오 디코더
    cuda::GpuMat gImg;
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(videoPath);

    // Detection용 Mat
    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;


    // fps 계산용
    int frameCnt = 0;
    int nPrevSec = -1;

    SYSTEMTIME stNow;

    // Yolo
    //auto net = cv::dnn::readNetFromDarknet("D:/test/352.cfg", "D:/test/352.weights");

    //yolov5
    auto net = cv::dnn::readNetFromONNX("../../Runtime/config/smart_vehicle_640_640.onnx");
    std::vector<std::string> class_names;
    class_names.push_back("Car"); class_names.push_back("MotorCycle"); class_names.push_back("Bicycle"); class_names.push_back("Person"); class_names.push_back("PM");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    auto output_names = net.getUnconnectedOutLayersNames();

    // 루프 시작
    for (;;)
    {
        if (!d_reader->nextFrame(gImg))
            break;

        // fps 계산용
        frameCnt++;

        // gpu mat 상태로 연산
        int imageSize = 352;
        cv::cuda::cvtColor(gImg, gImg, COLOR_RGBA2RGB);
        cv::cuda::resize(gImg, gImg, Size(imageSize, imageSize));

        // 일반 Mat로 변환
        gImg.download(frame);

        // YOLO
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255, cv::Size(imageSize, imageSize), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);
        net.forward(detections, output_names);

        // 01 ----------------------------------------
        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
            }
        }

        // 이미지 출력
        //resize(frame, frame, Size(160, 160));
        if (bimshow == 1)
        {
            imshow(to_string(threadNum), frame);
        }

        // 시간 측정 종료 및 출력
        ::GetLocalTime(&stNow);
        if (stNow.wSecond != nPrevSec)
        {
            nPrevSec = stNow.wSecond;
            cout << frameCnt << " / ";

            frameCnt = 0;
        }
        waitKey(1);
        Sleep(30);
    }
}


int main(int argc, char** argv)
{
    string videoPath = "rtsp://210.99.70.120:1935/live/cctv001.stream";
    //string videoPath = "../Runtime/Data/test.mp4";


    // 총 쓰레드 수
    int num_threads = 8;

    // 그래픽 카드당 처리 채널 수
    int channel4grapich = 8;

    int bimshow = 1;

    if (argc == 5)
    {
        num_threads = atoi(argv[1]);
        channel4grapich = atoi(argv[2]);
        videoPath = argv[3];
        bimshow = atoi(argv[4]);
    }

    // 쓰레드 벡터
    vector<std::thread*> thread_Ptr_vec;


    // 쓰레드 생성
    for (int i = 0; i < num_threads; i++)
    {
        thread_Ptr_vec.push_back(new std::thread(gpuVideoThread, i, channel4grapich, videoPath, bimshow));
    }

    for (int i = 0; i < num_threads; i++)
    {
        thread_Ptr_vec[i]->join();
    }


    return 0;
}
