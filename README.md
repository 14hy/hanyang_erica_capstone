# Artificial Intelligent Recycling Trashbox

This is capstone project in Hanyang University ERICA campus.  
Our project is about automaticly classifying which category a trash is in.  
We used 4 categories which are can, plastic, glass, and extra. But, in exhibition, we used only 3 categories whihout extra category.


# Demo

https://wayexists.notion.site/Capstone-Design-Project-00b40a81738d427d9d6b0d7e627b451b


# Process

I used ROS(Robot Operating System) for client system for raspberry pi.  
I used GCP for deep learning server.  
The overall process is as following.

![capstone](https://user-images.githubusercontent.com/26874750/59034404-f3a26b80-88a5-11e9-8edc-56cab6ffcc4d.PNG)

# Modules

### ai

Run in GCP virtual machine. This implements core deep learning part to classify category of recycle trash. We implemented deep neural net in pytorch.

Following picture is neural net for detecting trash whether thing is a real trash or just hand of human.    
![capstone2](https://user-images.githubusercontent.com/26874750/59036904-256a0100-88ab-11e9-9031-fa3a8050c61a.PNG)


And, this picture is our main neural network for classification.  
![capstone3](https://user-images.githubusercontent.com/26874750/59037385-1cc5fa80-88ac-11e9-85c2-d8a97389adae.PNG)


### client

This module is just for testing whether the transmitting image to GCP server is well done.


### helper

We must have collected data from scratch. So, we decided to develop a helper software for collecting data easily. This module is that.


### rpi/ros

This module is run in raspberry pi. In ros folder, there are 3 module which are ROS nodes.


### server

This module is run in GCP for opening server socket. This module have dependency on ai module.
