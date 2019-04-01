clear all
close all
clc

%gt_pa = './data/highway/groundtruth/';
%gt_ft = 'png';


%[files data] = loadData_plus(gt_pa, gt_ft);

gtim = double(imread('bear02_0458_gt.png'));
fgim = double(imread('modified_pca_binary_mask_bear02_0458.jpg.png'));


[TP FP FN TN] = evaluation_entry(fgim,gtim);

Re = TP/(TP + FN)
Pr = TP / (TP + FP)
Fm = (2*Pr*Re)/(Pr + Re)
