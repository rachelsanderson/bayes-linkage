#!/bin/bash
#SBATCH --nodes=1
#BATCH --ntasks-per-node=1
#SBATCH --time=1:01:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=rachelsa@princeton.edu

./my_app
