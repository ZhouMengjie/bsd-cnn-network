% statistic BSDs distribution from different cities
clear all
addpath('features');
city_list = {'bath';'bristol';'cambridge';'cheltenham';'coventry';'derby';'glasgow';...
    'leeds';'liverpool';'livingston';'manchester';'newcastle';'norwich';'sheffield';...
    'southampton';'plymouth';'preston';'wakefield';'walsall';'wolverhampton';'york';...
    'nottingham';'leicester';'cardiff';'belfast';'brighton';'aberdeen';'inverness';...
    'durham';'birmingham';'dublin';'lyon';'helsinki';'berlin';'amsterdam';'madrid';...
    'vienna';'athens';'prague';'milan';'miami';'dallas';'atlanta';'chicago';'columbus';...
    'calgary';'edmonton';'ottawa';'montreal';'vancouver'};
dataset = struct();
dataset_general = struct();
sum = 0;
sum_gaps = 0;
sum_nongaps = 0;
sum_juncs = 0;
sum_nonjuncs = 0;
for i=1:size(city_list, 1)
    data = city_list{i,1};
    load(['features/','BSD','/','BSD','_', data,'_10_19','.mat']);
    dataset(i).city = data;
    dataset(i).total = size(routes,2)*2;
    
    dataset(i).front_juncs = 0;
    dataset(i).front_nonjuncs = 0;
    
    dataset(i).back_juncs = 0;
    dataset(i).back_nonjuncs = 0;
    
    dataset(i).right_gaps = 0;
    dataset(i).right_nongaps = 0;
    
    dataset(i).left_gaps = 0;
    dataset(i).left_nongaps = 0;
    
    
    for j=1:length(routes)
        desc = routes(j).BSDs;
        if desc(1) == 1
            dataset(i).front_juncs = dataset(i).front_juncs+1;
        else
            dataset(i).front_nonjuncs =  dataset(i).front_nonjuncs+1;
        end
        
        if desc(3) == 1
            dataset(i).back_juncs = dataset(i).back_juncs+1;
        else
            dataset(i).back_nonjuncs =  dataset(i).back_nonjuncs+1;
        end
        
        if desc(2) == 1
            dataset(i).right_gaps = dataset(i).right_gaps+1;
        else
            dataset(i).right_nongaps =  dataset(i).right_nongaps+1;
        end
        
        if desc(4) == 1
            dataset(i).left_gaps = dataset(i).left_gaps+1;
        else
            dataset(i).left_nongaps =  dataset(i).left_nongaps+1;
        end               
    end
    dataset_general(i).city = data;
    dataset_general(i).total = size(routes,2)*2;
    dataset_general(i).juncs = dataset(i).front_juncs + dataset(i).back_juncs;
    dataset_general(i).nonjuncs = dataset(i).front_nonjuncs + dataset(i).back_nonjuncs;
    dataset_general(i).gaps = dataset(i).left_gaps + dataset(i).right_gaps;
    dataset_general(i).nongaps = dataset(i).left_nongaps + dataset(i).right_nongaps;
    sum = dataset(i).total + sum; 
    sum_juncs = sum_juncs + dataset_general(i).juncs;
    sum_nonjuncs = sum_nonjuncs + dataset_general(i).nonjuncs;
    sum_gaps = sum_gaps + dataset_general(i).gaps;
    sum_nongaps = sum_nongaps + dataset_general(i).nongaps;
    
end
save('dataset.mat',  '-v7.3')
p = ['there are', ' ', num2str(sum), ' in total for each category'];
disp(p)

