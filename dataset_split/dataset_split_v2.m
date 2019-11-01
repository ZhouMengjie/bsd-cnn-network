% dataset split_v2 (training, validation, testing)
% total - 464064
% training - junctions:190000 * 2, gaps: 200000 * 2
% test - junctions:14393 * 2, gaps: 12155 * 2
clear all
load('dataset.mat');
train_junctions = 190000;
train_gaps = 200000;

test_junctions = 14393;
test_gaps = 12155;

% just for re-check
junctions_train_total = 0;
nonjunctions_train_total = 0;
junctions_test_total = 0;
nonjunctions_test_total = 0;

gaps_train_total = 0;
nongaps_train_total = 0;
gaps_test_total = 0;
nongaps_test_total = 0;

set_split = struct();
for i=1:length(dataset_general)
    set_split(i).city = dataset_general(i).city;
    set_split(i).junctions_train = ceil(dataset_general(i).juncs / sum_juncs * train_junctions);   
    set_split(i).nonjunctions_train = ceil(dataset_general(i).nonjuncs / sum_nonjuncs * train_junctions);    
    set_split(i).junctions_test = ceil(dataset_general(i).juncs / sum_juncs * test_junctions);   
    set_split(i).nonjunctions_test = ceil(dataset_general(i).nonjuncs / sum_nonjuncs * test_junctions);
    
    set_split(i).gaps_train = ceil(dataset_general(i).juncs / sum_juncs * train_gaps);   
    set_split(i).nongaps_train = ceil(dataset_general(i).nonjuncs / sum_nonjuncs * train_gaps);    
    set_split(i).gaps_test = ceil(dataset_general(i).juncs / sum_juncs * test_gaps);   
    set_split(i).nongaps_test = ceil(dataset_general(i).nonjuncs / sum_nonjuncs * test_gaps); 
    
    junctions_train_total = junctions_train_total + set_split(i).junctions_train;
    nonjunctions_train_total = nonjunctions_train_total + set_split(i).nonjunctions_train;
    junctions_test_total = junctions_test_total + set_split(i).junctions_test;
    nonjunctions_test_total = nonjunctions_test_total + set_split(i).nonjunctions_test;
    
    gaps_train_total = gaps_train_total + set_split(i).gaps_train;
    nongaps_train_total = nongaps_train_total + set_split(i).nongaps_train;
    gaps_test_total =  gaps_test_total + set_split(i).gaps_test;
    nongaps_test_total = nongaps_test_total + set_split(i).nongaps_test;  
    
    junctions_total = junctions_train_total + nonjunctions_train_total + junctions_test_total + nonjunctions_test_total;
    gaps_total = gaps_train_total + nongaps_train_total + gaps_test_total + nongaps_test_total;
end
save('dataset_split_v2.mat',  '-v7.3')