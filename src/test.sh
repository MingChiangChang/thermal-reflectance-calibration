#!/bin/sh
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -d 800 -p 30 -pre TEST 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 250 -p 15 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 250 -p 20 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 250 -p 25 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 500 -p 15 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 500 -p 20 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 500 -p 25 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 1000 -p 15 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 1000 -p 20 
python ThermalReflectance.py -n 5 -pmin 0 -20 -pmax 0 -20 -pre TEST -d 1000 -p 25 
