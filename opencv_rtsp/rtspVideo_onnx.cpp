//C++ header file 
#include <iostream>

//opencv header file include
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

//#define VIDEO_PATH "C:\\Users\\Minji\\source\\repos\\OpenCV_Test\\OpenCV_Test\\vtest.avi"
#define WEBCAM_NUM 0
////https://www.wowza.com/html/mobile.html
#define RTSP_URL "rtsp://210.99.70.120:1935/live/cctv002.stream"

#define VIDEO_WINDOW_NAME "video"

//project main function
int main(int argc, char** argv) {

	//opencv videocapture class
	//������, ��ķ, RTSP ������ �ҷ��� �� �ִ�.
	cv::VideoCapture videoCapture(RTSP_URL);

	//������ �ҷ����� ���� ��
	if (!videoCapture.isOpened()) {
		std::cout << "Can't open video !!!" << std::endl;
		return -1;
	}

	//OpenCV Mat class
	cv::Mat videoFrame;


	float videoFPS = videoCapture.get(cv::CAP_PROP_FPS);
	int videoWidth = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
	int videoHeight = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::cout << "Video Info" << std::endl;
	std::cout << "video FPS : " << videoFPS << std::endl;
	std::cout << "video width : " << videoWidth << std::endl;
	std::cout << "video height : " << videoHeight << std::endl;

	//�̹����� window�� �����Ͽ� �����ݴϴ�.
	cv::namedWindow(VIDEO_WINDOW_NAME);

	//video ��� ����
	while (true) {
		//VideoCapture�� ���� �������� �޾ƿ´�
		videoCapture >> videoFrame;

		//ĸ�� ȭ���� ���� ���� Video�� ���� ���
		if (videoFrame.empty()) {
			std::cout << "Video END" << std::endl;
		}

		cv::imshow(VIDEO_WINDOW_NAME, videoFrame);

		//'ESC'Ű�� ������ ����ȴ�.
		//FPS�� �̿��Ͽ� ���� ��� �ӵ��� �����Ͽ��ش�.
		//1000�� 1��, fps�� 1�ʴ� ������ � -> ������ �Ѱ��� 1/n�� (Ű �Է� ��� �ð�)
		//27�� escŰ
		if (cv::waitKey(1000 / videoFPS) == 27) {
			std::cout << "Stop Video" << std::endl;
			break;
		}
	}


	//�����Ͽ��� �����츦 �����մϴ�.
	cv::destroyWindow(VIDEO_WINDOW_NAME);

	//�Ʒ��� �Լ��� ����ϸ�, ����ϰ� �ִ� ������ ���θ� �����մϴ�.
	//cv::destroyAllWindows();

	return 0;
}
