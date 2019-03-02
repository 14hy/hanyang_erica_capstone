#include "ros/ros.h"
#include "std_srvs/SetBool.h"
#include "usb_camera/UsbCamera.hpp"
#include "usb_camera/usb_camera_node.hpp"

UsbCamera* usbcam;

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "usb_camera_node");
	ros::NodeHandle nh;

	/*
	ros::ServiceServer serv_done = nh.advertiseService(
		"done",
		serv_done_handler
	);
	ros::ServiceServer serv_wait = nh.advertiseService(
		"wait",
		serv_wait_handler
	);
	*/

	usbcam = new UsbCamera(nh);
	usbcam->init();

	ros::Rate rate(20);

	while (ros::ok()) {
		usbcam->compute();

		rate.sleep();
		ros::spinOnce();
	}

	usbcam->finalize();
	delete usbcam;

	return 0;
}

bool serv_done_handler(std_srvs::SetBool::Request& req,
		  std_srvs::SetBool::Response& res)
{
	if (usbcam->isWaiting() && req.data)
		usbcam->resume();

	res.success = true;
	res.message = "success";

	return true;
}

bool serv_wait_handler(std_srvs::SetBool::Request& req,
		       std_srvs::SetBool::Response& res)
{
	if (req.data)
		usbcam->waitForDone();

	res.success = true;
	res.message = "success";

	return true;
}
