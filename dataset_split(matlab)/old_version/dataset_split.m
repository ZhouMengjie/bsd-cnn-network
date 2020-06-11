% dataset split (training, validation, testing)
% training - 95%
% validation - 2.5%
% testing - 2.5%
clear all
load('dataset.mat','dataset_general');
set_split = struct();
train_per = 0.95;
val_per = 0.025;
test_per = 0.025;

junctions_train_total = 0;
junctions_val_total = 0;
junctions_test_total = 0;

gaps_train_total = 0;
gaps_val_total = 0;
gaps_test_total = 0;

for i=1:length(dataset_general)
    set_split(i).city = dataset_general(i).city;
    % for junctions
    train_num_juncs = round(dataset_general(i).juncs * 0.95);
    train_num_nonjuncs = round(dataset_general(i).nonjuncs * 0.95);
    val_num_juncs = round(dataset_general(i).juncs * 0.025);
    val_num_nonjuncs = round(dataset_general(i).nonjuncs * 0.025);
    test_num_juncs = round(dataset_general(i).juncs * 0.025);
    test_num_nonjuncs = round(dataset_general(i).nonjuncs * 0.025);
    
    set_split(i).junctions_train = min(train_num_juncs, train_num_nonjuncs);
    set_split(i).nonjunctions_train = set_split(i).junctions_train;
    
    set_split(i).junctions_val = min(val_num_juncs, val_num_nonjuncs);
    set_split(i).nonjunctions_val = set_split(i).junctions_val;
    
    set_split(i).junctions_test = min(test_num_juncs, test_num_nonjuncs);
    set_split(i).nonjunctions_test = set_split(i).junctions_test;
    
    % for gaps
    train_num_gaps = round(dataset_general(i).gaps * 0.95);
    train_num_nongaps = round(dataset_general(i).nongaps * 0.95);
    val_num_gaps = round(dataset_general(i).gaps * 0.025);
    val_num_nongaps = round(dataset_general(i).nongaps * 0.025);
    test_num_gaps = round(dataset_general(i).gaps * 0.025);
    test_num_nongaps = round(dataset_general(i).nongaps * 0.025);
    
    set_split(i).gaps_train = min(train_num_gaps, train_num_nongaps);
    set_split(i).nongaps_train = set_split(i).gaps_train;
    
    set_split(i).gaps_val = min(val_num_gaps, val_num_nongaps);
    set_split(i).nongaps_val = set_split(i).gaps_val;
    
    set_split(i).gaps_test = min(test_num_gaps, test_num_nongaps);
    set_split(i).nongaps_test = set_split(i).gaps_test;
    
    junctions_train_total = junctions_train_total + set_split(i).junctions_train * 2;
    junctions_val_total = junctions_val_total + set_split(i).junctions_val * 2;
    junctions_test_total = junctions_test_total + set_split(i).junctions_test * 2;
    
    gaps_train_total = gaps_train_total + set_split(i).gaps_train * 2;
    gaps_val_total = gaps_val_total + set_split(i).gaps_val * 2;
    gaps_test_total = gaps_test_total + set_split(i).gaps_test * 2;    
end
save('dataset_split.mat',  '-v7.3')

p1 = ['there are', ' ', num2str(junctions_train_total), ' in total for junctions traning set'];
disp(p1)
p2 = ['there are', ' ', num2str(junctions_val_total), ' in total for junctions validation set'];
disp(p2)
p3 = ['there are', ' ', num2str(junctions_test_total), ' in total for junctions testing set'];
disp(p3)
p4 = ['there are', ' ', num2str(gaps_train_total), ' in total for gaps traning set'];
disp(p4)
p5 = ['there are', ' ', num2str(gaps_val_total), ' in total for gaps validation set'];
disp(p5)
p6 = ['there are', ' ', num2str(gaps_test_total), ' in total for gaps testing set'];
disp(p6)
