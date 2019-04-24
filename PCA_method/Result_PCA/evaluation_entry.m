function [TP FP FN TN] = evaluation_detection(fgimage,truth)

fgP_index = fgimage == 255;
fgN_index = fgimage == 0;

trP_index = truth == 255;
trN_index = truth == 0;

index = fgP_index & trP_index;
TP = sum(sum(index));

index = fgP_index & trN_index;
FP = sum(sum(index));

index = fgN_index & trN_index;
TN = sum(sum(index));

index = fgN_index & trP_index;
FN = sum(sum(index));



