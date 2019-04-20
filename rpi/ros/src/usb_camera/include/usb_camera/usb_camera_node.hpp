#ifndef __USB_CAMERA_HPP__
#define __USB_CAMERA_HPP__

#include "std_srvs/SetBool.h"

bool serv_done_handler(std_srvs::SetBool::Request& req,
		       std_srvs::SetBool::Response& res);
bool serv_wait_handler(std_srvs::SetBool::Request& req,
		       std_srvs::SetBool::Response& res);

#endif
