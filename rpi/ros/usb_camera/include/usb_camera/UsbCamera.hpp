#ifndef __USB_CAMERA_HPP__
#define __USB_CAMERA_HPP

#include "ros/ros.h"
#include "opencv2/opencv.hpp"

class UsbCamera {
public:
	UsbCamera(ros::NodeHandle& handle);
	virtual ~UsbCamera();

	void init();
	void finalize();
	void compute();
	void resume();
	void waitForDone();
	bool isWaiting();

private:
	bool stop;
	bool wait;

	ros::NodeHandle nh;
	ros::Publisher pub;
	cv::VideoCapture* cap;
	cv::Mat initial_frame;

	bool isValuableFrame(cv::Mat& frame);
	void publish(cv::Mat& frame);
};

#endif
