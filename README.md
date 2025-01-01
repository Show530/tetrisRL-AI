# Tetris RL AI training
This is a Tetris reinforcement learning code capable of training a model. To run the code, first open a terminal with the code and run the following commands:

# If using a Mac or Linux:
Run the following command to complie the code
```bash
javac -cp "./lib/*:." @tetris.srcs
```
Run the following command to execute the code
```bash 
java -cp "./lib/*:." edu.bu.tetris.Main -p 5000 -t 200 -v 100 -g 0.99 -n 1e-5
```

# If using a Windows:
Run the following command to complie the code
```bash
javac -cp "./lib/*;." @tetris.srcs
```
Run the following command to execute the code
```bash
java -cp "./lib/*;." edu.bu.tetris.Main -p 5000 -t 200 -v 100 -g 0.99 -n 1e-5
```

# General other points:
Once the code has been complied, adding a -h to the execution code without any other flags will show the flags available and what they do.
Points are earned for cleared lines, perfect clears, a tetris, and t-spins.
