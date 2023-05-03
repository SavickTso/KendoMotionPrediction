# How to use docker
For full tutorial, open a terminal by `ctrl+alt+t`, and type:
  `docker run -dp 80:80 docker/getting-started`
Open a web browser and access:
  `http://localhost`

## Check images we got now
`sudo docker ps` 

## To pull an image
`sudo docker pull {YOUR_IMAGE_NAME}`

## To create a container from you image
`sudo docker run -it --name {YOUR_CONTAINER_NAME} --rm {YOUR_IMAGE_NAME:tag}`

**If you want to start GUI** 
Add these two parameters `-e DISPLAY=$DISPLAY`, `-v /tmp/.X11-unix:/tmp/.X11-unix`, like the command below.
`sudo docker run -it --name {YOUR_CONTAINER_NAME} --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix {YOUR_IMAGE_NAME:tag}`

## Communicate between multiple containers
Add parameter `--network=host`, like the command below.
`sudo docker run -it --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm osrf/ros:noetic-desktop-full`

## To save your current container as a new image
`sudo docker commit {CONTAINER_ID} {NEW_IMAGE_NAME}`

use `sudo docker container ls` to check container ID.

HAVE FUN!
