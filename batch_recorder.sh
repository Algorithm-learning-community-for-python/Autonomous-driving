#!/bin/bash
# My first script



# FROM OLD ITERATION
#cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..11} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "1"
	echo "------------------------     Iteration $i out of 61     ------------------------------------"
	echo "cars_noise_random_weather"							
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/cars_noise_random_weather_2 -c 1 -t 1 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..4} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "1"
	echo "------------------------     Iteration $i out of 61     ------------------------------------"
	echo "cars_noise_random_weather"							
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/cars_noise_random_weather_2 -c 1 -t 1 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done
: '
#cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..21} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "2"
	echo "------------------------     Iteration $i out of 21     ------------------------------------"
	echo "cars_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/cars_noise_random_weather -c 1 -t 1 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#cars_no_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..21} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "3"
	echo "------------------------     Iteration $i out of 61     ------------------------------------"
	echo "cars_no_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/cars_no_noise_random_weather -c 1 -t 1 -w 1 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#cars_no_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..21} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "4"
	echo "------------------------     Iteration $i out of 21     ------------------------------------"
	echo "cars_no_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/cars_no_noise_random_weather -c 1 -t 1 -w 1 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done


# NO CARS NO NOISE in CLOUDYNOON
sudo echo "Fetching root priveliged"
for i in {1..31} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "5"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_no_noise_cloudynoon"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/no_cars_no_noise_cloudynoon -c 0 -t 0 -w 0 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

sudo echo "Fetching root priveliged"
for i in {1..11} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "6"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_no_noise_cloudynoon"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/no_cars_no_noise_cloudynoon -c 0 -t 0 -w 0 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done
#no_cars_noise_cloudynoon 
sudo echo "Fetching root priveliged"
for i in {1..31} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "7"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_noise_cloudynoon"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/no_cars_noise_cloudynoon -c 0 -t 0 -w 0 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

sudo echo "Fetching root priveliged"
for i in {1..11} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "8"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_noise_cloudynoon"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/no_cars_noise_cloudynoon -c 0 -t 0 -w 0 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done
#no_cars_no_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..31} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "9"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_no_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/no_cars_no_noise_random_weather -c 0 -t 0 -w 1 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done
sudo echo "Fetching root priveliged"
for i in {1..11} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "10"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_no_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/no_cars_no_noise_random_weather -c 0 -t 0 -w 1 -n 0 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#no_cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..31} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "11"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/no_cars_noise_random_weather -c 0 -t 0 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#no_cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..11} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "12"
	echo "------------------------     Iteration $i out of 30     ------------------------------------"
	echo "no_cars_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/no_cars_noise_random_weather -c 0 -t 0 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..61} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "15"
	echo "------------------------     Iteration $i out of 61     ------------------------------------"
	echo "cars_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Training_data/cars_noise_random_weather -c 1 -t 1 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done

#cars_noise_random_weather
sudo echo "Fetching root priveliged"
for i in {1..21} 
do
	echo "Starting Carla"
	DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=30 & 
	pid=$!	
	sleep 10

	echo "Carla PID: $pid"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "16"
	echo "------------------------     Iteration $i out of 61     ------------------------------------"
	echo "cars_noise_random_weather"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "#############################################################################################"
	echo "starting recorder"
	python2 recorder_trimmed.py --path Validation_data/cars_noise_random_weather -c 1 -t 1 -w 1 -n 1 #> output_log/stdoutrecorder$i.txt 2> output_log/stderrecorder$i.txt & 
	echo "killing Carla"
	kill -SIGINT $pid
	sudo fuser -k -n tcp 2000
done
'