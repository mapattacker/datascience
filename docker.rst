Docker
=================

Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, 
and ship it all out as one package. They allow a modular construction of an application architecture, or microservice in short.
Docker is a popular tool designed to make it easier to create, deploy, and run applications by using containers.

Preprocessing scripts and models can be created as a docker image snapshot, and launched as a container in production.

https://runnable.com/docker/python/dockerize-your-python-application
https://docs.docker.com/get-started/part2/

Creating Images
--------------------
To start of a new project, create a new folder. This should only contain your docker file and related python files.

Dockerfile
***********
A dockerfile, is a file without extension type. It contains commands to tell docker what are the steps to do to
create an image. It consists of instructions & arguments.

.. figure:: images/docker_build1.png
    :width: 600px
    :align: center

    from Udemy's Docker for the Absolute Beginner - Hands On

The commands run sequentially when building the image, also known as a layered architecture. 
Each layer is cached, such that when any layer fails and is fixed, rebuilding it will start from the last built layer.

.. figure:: images/docker_build2.png
    :width: 600px
    :align: center

    from Udemy's Docker for the Absolute Beginner - Hands On

 * ``FROM`` tells Docker which image you base your image on (eg, Python 3 or continuumio/miniconda3).
 * ``RUN`` tells Docker which additional commands to execute.
 * ``CMD`` tells Docker to execute the command when the image loads.

.. code::

    # Use an official Python runtime as a parent image
    FROM python:2.7-slim

    # Set the working directory to /app
    WORKDIR /app

    # Copy the current directory contents into the container at /app
    COPY . /app

    # Install any needed packages specified in requirements.txt
    RUN pip install --trusted-host pypi.python.org -r requirements.txt

    # Make port 80 available to the world outside this container
    EXPOSE 80

    # Define environment variable
    ENV NAME World

    # Run app.py when the container launches
    CMD ["python", "app.py"]


Build the Image
*******************
``docker build -t image-name .`` --(-t = tag the image as) build and name image

Push to Dockerhub
********************

``docker push image_name``

Commands
----------

Help
 * ``docker --help`` --list all base commands
 * ``docker COMMAND --help`` --list all options for a command

Create Image
 * ``docker build -t image_name .`` --(-t = tag the image as) build and name image, "." is the location of the dockerfile

Get Image from Docker Hub
 * ``docker pull image_name`` --pull image from dockerhub into docker
 * ``docker run image_name COMMAND`` --check if image in docker, if not pull & run image from dockerhub into docker. If no command is given, the container will stop running.
 * ``docker run image_name cat /etc/*release*`` --run image and print out the version of image

Other Run Commands
 * ``docker run Ubuntu:17.04`` --semicolon specifies the version (known as tags as listed in Dockerhub), else will pull the latest
 * ``docker run ubuntu`` vs ``docker run mmumshad/ubuntu`` --the first is an official image, the 2nd with the / is created by the community
 * ``docker run -d image_name`` --(-d = detach) docker runs in background, and you can continue typing other commands in the bash. Else need to open another terminal.
 * ``docker run -v /local/storage/folder:/image/data/folder mysql`` --(-v = volume mapping) all data will be destroyed if container is stopped

.. figure:: images/docker_cmd.PNG
    :scale: 100 %
    :align: center

    running docker with a command. each container has a unique container ID, container name, and their base image name

IPs & Ports
 * ``192.168.1.14`` --IP address of docker host
 * ``docker inspect container_id`` --dump of container info, as well as at the bottom, under Network, the internal IP address. to view server in web browser, enter the ip and the exposed port. eg. 172.17.0.2:8080
 * ``docker run -p 80:5000 image_name`` --(host_port:container_port) map host service port with the container port on docker host

See Images & Containers in Docker
 * ``docker images`` --see all installed docker images
 * ``docker ps`` --(ps = process status) show status of images which are running
 * ``docker ps -a`` --(-a = all) show status of all images including those that had exited

Start/Stop Containers
 * ``docker start container_name`` --run container
 * ``docker stop container_name`` --stop container from running, but container still lives in the disk
 * ``docker stop container_name1 container_name2`` --stop multiple container from running in a single line
 * ``docker stop container_id`` --stop container using the ID. There is no need to type the id in full, just the first few char suffices.

Remove Containers/Images
 * ``docker rm container_name`` --remove container from docker
 * ``docker rmi image_name`` --(rmi = remove image) from docker. must remove container b4 removing image.
 * ``docker -f rmi image_name`` --(-f = force) force remove image even if container is running

Execute Commands for Containers
 * ``docker exec container_nm COMMAND`` --execute a command within container