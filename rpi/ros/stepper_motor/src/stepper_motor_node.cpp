#include "ros/ros.h"
#include "wiringPi.h"

#include "stepper_motor/StepperMotor.hpp"
#include "stepper_motor/stepper_motor.hpp"

int main(int argc, char* argv[])
{
	setWiringPi();

	ros::init(argc, argv, "stepper_motor_node");
	ros::NodeHandle nh;

	StepperMotor motor(nh);
	motor.Setup();

	ros::Rate rate(10);

	while (ros::ok()) {
		rate.sleep();
		ros::spinOnce();
	}

	motor.Destroy();

	return 0;
}

void setWiringPi()
{
	wiringPiSetup();

	pinMode(MOTOR_ENABLE, OUTPUT);
	pinMode(LEFT_MOTOR_CLK, OUTPUT);
	pinMode(LEFT_MOTOR_DIR, OUTPUT);
	pinMode(RIGHT_MOTOR_CLK, OUTPUT);
	pinMode(RIGHT_MOTOR_DIR, OUTPUT);
}
