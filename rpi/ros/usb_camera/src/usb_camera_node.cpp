#include "ros/ros.h"
#include "std_srvs/SetBool.h"
#include "usb_camera/UsbCamera.hpp"
#include "usb_camera/usb_camera_node.hpp"

UsbCamera* usbcam;

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "usb_camera_node");
	ros::NodeHandle nh;

	usbcam = new UsbCamera(nh);
	usbcam->init();

	ros::Rate rate(10);

	while (ros::ok()) {
		usbcam->compute();

		rate.sleep();
		ros::spinOnce();
	}

	usbcam->finalize();
	delete usbcam;

	return 0;
}

