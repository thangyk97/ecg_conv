clear all; close all; clc;
load('C:\Users\thang\Documents\MATLAB\arrhythmia_database_mit\100m.mat');

% data = data(:,2);
% data = data(1:11700);

[qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(val(2,:),360,2);
