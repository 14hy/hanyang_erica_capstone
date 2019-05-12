#include "stepper_motor/StepperMotor.hpp"
#include "stepper_motor/stepper_motor.hpp"
#include "std_srvs/SetBool.h"
#include "wiringPi.h"
#include <thread>
#include <vector>

StepperMotor::StepperMotor(ros::NodeHandle _nh)
	: nh(_nh), is_ready(true)
{

}

StepperMotor::~StepperMotor()
{

}

void StepperMotor::Setup()
{
	this->sub = nh.subscribe("motor", 2,
			&StepperMotor::MotorCallback, this);
	this->serv_clnt = nh.serviceClient<std_srvs::SetBool>("motor_done");
}

void StepperMotor::Destroy()
{
	
}

void StepperMotor::MotorCallback(const std_msgs::Int32MultiArray::ConstPtr& ptr)
{
	if (this->is_ready) {
		this->is_ready = false;

		std::vector<int> data = ptr->data;

		ROS_INFO("Motor: %d", data[0]);
		ROS_INFO("Motor dir: %d", data[1]);
		ROS_INFO("Motor step: %d", data[2]);

		if (data[0] == BOX_MOTOR) {
			digitalWrite(BOX_MOTOR_ENABLE, HIGH);
			GoStep(box_motor1, data[1], data[2]);
			GoStep(box_motor2, data[1], data[2]);

			while (!this->is_ready);
			digitalWrite(BOX_MOTOR_ENABLE, LOW);
		}
		else if (data[0] == SUPPORT_MOTOR) {
			digitalWrite(SUPPORT_MOTOR_ENABLE, HIGH);
			GoStep(support_motor, data[1], data[2]);

			while (!this->is_ready);
			digitalWrite(SUPPORT_MOTOR_ENABLE, LOW);
		}

		std_srvs::SetBool request;
		request.request.data = true;

		this->serv_clnt.call(request);
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

	this->is_ready = true;
}
