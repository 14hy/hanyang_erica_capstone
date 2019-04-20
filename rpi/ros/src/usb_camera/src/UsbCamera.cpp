#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include "ros/ros.h"
#include "std_msgs/UInt8MultiArray.h"
#include "usb_camera/UsbCamera.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#define TIME_INTERVAL 250
#define HEIGHT 128
#define WIDTH 128
#define CHANNEL 3
#define SIZE (HEIGHT * WIDTH * CHANNEL)

#define THRESHOLD 50

typedef cv::Point3_<uint8_t> Pixel;

UsbCamera::UsbCamera(ros::NodeHandle& handle)
	: nh(handle), stop(false), cap(NULL), wait(false)
{

}

UsbCamera::~UsbCamera()
{
	delete cap;
}

void UsbCamera::init()
{
	pub = nh.advertise<std_msgs::UInt8MultiArray>(
		"image_data",
		0
	);

	cap = new cv::VideoCapture(0);

	cv::Mat temp;
	*cap >> temp;

	cv::resize(temp, initial_frame, cv::Size(HEIGHT, WIDTH));

	ROS_INFO("Thread is starting...");
}

void UsbCamera::finalize()
{
	stop = true;
}

void UsbCamera::resume()
{
	ROS_INFO("Usb camera node resumed.");
	wait = false;
}

bool UsbCamera::isWaiting()
{
	return wait;
}

void UsbCamera::compute()
{
	static int time_count = 0;
	static int send_count = 0;

	if (!cap->isOpened()) {
		ROS_ERROR("CANNOT open camera");
		finalize();
		return;
	}

	time_count += 1;

	if (time_count != 1) return;

	time_count = 0;

	cv::Mat frame;
	cv::Mat bgrFrame;
	//cv::Mat rgbFrame;
	cv::Mat temp;

	*cap >> frame;
	cv::resize(frame, bgrFrame, cv::Size(HEIGHT, WIDTH));
	//cv::cvtColor(temp, rgbFrame, cv::COLOR_BGRA2BGR);

	if (!wait && (send_count > 0 || isValuableFrame(bgrFrame))) {
		//cv::cvtColor(bgrFrame, rgbFrame, CV_BGR2RGB);
		publish(bgrFrame);
		send_count += 1;
		if (send_count == 8) {
			send_count = 0;
			time_count = -TIME_INTERVAL;
			//wait = true;
		}
		//wait = true;
	}

	//rgbFrame.copyTo(previous_frame);
}

void UsbCamera::publish(cv::Mat& frame)
{
	std_msgs::UInt8MultiArray msg;

	msg.data.resize(SIZE);
	memcpy(msg.data.data(), frame.data, SIZE);

	ROS_INFO("Publishing...");
	pub.publish(msg);
}

void UsbCamera::waitForDone()
{
	ROS_INFO("Usb camera will be wait for done");
	wait = true;
}

bool UsbCamera::isValuableFrame(cv::Mat& frame)
{

	cv::Mat curFrame;
	frame.copyTo(curFrame);

	/*
	preFrame.forEach<Pixel>([=](Pixel& p, const int* pos) -> void {
		p.x /= 255;
		p.y /= 255;
		p.z /= 255;
	});
	curFrame.forEach<Pixel>([=](Pixel& p, const int* pos) -> void {
		p.x /= 255;
		p.y /= 255;
		p.z /= 255;
	});
	*/

	cv::Mat res1, res2;
	cv::subtract(curFrame, initial_frame, res1);
	cv::subtract(initial_frame, curFrame, res2);

	long distance = 0;
	for (int h = 0; h < HEIGHT; h++) {
		for (int w = 0; w < WIDTH; w++) {
			uint8_t x1 = res1.at<cv::Vec3b>(h, w)[0];
			uint8_t y1 = res1.at<cv::Vec3b>(h, w)[1];
			uint8_t z1 = res1.at<cv::Vec3b>(h, w)[2];

			uint8_t x2 = res2.at<cv::Vec3b>(h, w)[0];
			uint8_t y2 = res2.at<cv::Vec3b>(h, w)[1];
			uint8_t z2 = res2.at<cv::Vec3b>(h, w)[2];

			uint8_t x = x1 > x2 ? x1 : x2;
			uint8_t y = y1 > y2 ? y1 : y2;
			uint8_t z = z1 > z2 ? z1 : z2;

			distance += ((long)x * (long)x);
			distance += ((long)y * (long)y);
			distance += ((long)z * (long)z);
		}
	}

	distance /= (HEIGHT * WIDTH * CHANNEL);
	ROS_INFO("DISTANCE: %ld", distance);

	if (distance >= (long)THRESHOLD)
		return true;
	else
		return false;
}
