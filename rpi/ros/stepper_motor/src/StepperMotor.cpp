#include "stepper_motor/StepperMotor.hpp"
#include "stepper_motor/stepper_motor.hpp"
#include "wiringPi.h"
#include <thread>

StepperMotor::StepperMotor(ros::NodeHandle _nh)
	: nh(_nh)
{

}

StepperMotor::~StepperMotor()
{

}

void StepperMotor::Setup()
{
	this->sub_box = nh.subscribe("box_motor_command", 0,
			&StepperMotor::BoxMotorCallback, this);
	this->sub_support = nh.subscribe("support_motor_command", 0,
			&StepperMotor::SupportMotorCallback, this);

}

void StepperMotor::Destroy()
{
	
}

void StepperMotor::SupportMotorCallback(const std_msgs::Int32::ConstPtr& ptr)
{
	int data = ptr->data;

	ROS_INFO("SUPPORT motor input: %d", data);

	if (data == 0) { // default state
		GoStep(support_motor, FORWARD, 500);
	}
	else { // open the door!
		GoStep(support_motor, BACKWARD, 500);
	}
}

void StepperMotor::BoxMotorCallback(const std_msgs::Int32::ConstPtr& ptr)
{
	int data = ptr->data;

	ROS_INFO("BOX motor input: %d", data);

	switch (data) {
		case 0: // No category. default state
			//GoStep(0);
			break;
		case 1: // plastic
			GoStep(box_motor, FORWARD, 1000);
			break;
		case 2: // can
			GoStep(box_motor, FORWARD, 500);
			break;
		case 3: // glass
			GoStep(box_motor, BACKWARD, 500);
			break;
		case 4: // extra
			GoStep(box_motor, BACKWARD, 1000);
			break;
	}
}

void StepperMotor::GoStep(class MotorInfo const& info, int dir, int steps)
{
	int step = (dir == BACKWARD ? -steps : steps);
	std::thread th(&StepperMotor::Step, this, info, step);
	th.detach();
}

void StepperMotor::Step(class MotorInfo const& info, int steps)
{
	int motor_dir = info.motor_dir;
	int motor_clk = info.motor_clk;

	digitalWrite(MOTOR_ENABLE, HIGH);

	if (steps > 0) {
		digitalWrite(motor_dir, HIGH);
		
		int motor_clock = this->initial_motor_clock;

		int i = 0;

		for (; i < steps/2; i++) {
			digitalWrite(motor_clk, HIGH);
			delayMicroseconds(motor_clock);
			digitalWrite(motor_clk, LOW);
			delayMicroseconds(motor_clock);

			//motor_clock -= this->motor_speed_up;
			//if (motor_clock < this->min_motor_clock)
			//	motor_clock = this->min_motor_clock;
		}
		for (; i < steps; i++) {
			digitalWrite(motor_clk, HIGH);
			delayMicroseconds(motor_clock);
			digitalWrite(motor_clk, LOW);
			delayMicroseconds(motor_clock);

			//motor_clock += this->motor_speed_up;
			//if (motor_clock > this->initial_motor_clock)
			//	motor_clock = this->initial_motor_clock;
		}
	}
	else if (steps < 0) {
		digitalWrite(motor_dir, LOW);

		int motor_clock = initial_motor_clock;

		int i = 0;

		for (; i < -steps/2; i++) {
			digitalWrite(motor_clk, HIGH);
			delayMicroseconds(motor_clock);
			digitalWrite(motor_clk, LOW);
			delayMicroseconds(motor_clock);

			//motor_clock -= this->motor_speed_up;
			//if (motor_clock < this->min_motor_clock)
			//	motor_clock = this->min_motor_clock;
		}
		for (; i < -steps; i++) {
			digitalWrite(motor_clk, HIGH);
			delayMicroseconds(motor_clock);
			digitalWrite(motor_clk, LOW);
			delayMicroseconds(motor_clock);

			//motor_clock += this->motor_speed_up;
			//if (motor_clock > this->initial_motor_clock)
			//	motor_clock = this->initial_motor_clock;
		}
	}

	digitalWrite(MOTOR_ENABLE, LOW);
}
