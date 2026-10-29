# physical-ai-waste-sorting-robot
AI-powered autonomous waste sorting robot using PyBullet and A* path planning
# AI-Powered Smart Waste Sorting Robot

Physical AI Application Course - Final Project

## Overview
An AI-powered waste sorting robot developed using
Google Colab, Python, and PyBullet.

The robot classifies waste items and autonomously
transports them to designated recycling areas.

## Features
- AI-based waste classification using Random Forest
- Three waste categories: plastic, paper, and metal
- A* path planning for obstacle avoidance
- Autonomous waste collection and transportation
- Robot simulation using PyBullet

## Technologies
- Python
- Google Colab
- PyBullet
- scikit-learn
- A* Path Planning

## How to Run
1. Open the Jupyter Notebook in Google Colab.
2. Install the required Python packages.
3. Run all cells.
4. View the generated simulation video.

## Development Process
The initial implementation successfully classified
and collected waste but moved through other objects.

The improved version introduces A* path planning
to avoid uncollected waste during movement.

## Limitations
Waste classification primarily uses color features.
Obstacle positions are obtained from simulation data.
Robot collection is implemented using simplified
position control.
