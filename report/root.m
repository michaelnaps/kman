%% clean workspace
clc;clear;
close all;


%% path environments
addpath ./Data
addpath ./DataFunctions
addpath ./KoopFunctions
addpath ./PlotFunctions
addpath ./SphereWorld


%% initialize environment variables
% load world environments (including obstacles)
load sphereworld_minimal

% load pre-calculated Koopman operators
load K_21x21_learned
K_ls = K;

load K_21x21_derived
K_an = K;