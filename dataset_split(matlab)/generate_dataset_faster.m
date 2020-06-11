% generate dataset
clear all
addpath(genpath(pwd));
% addpath('features');
% addpath('images');
city = 'pittsburgh';
dataset = 'cmu5k'; 
outfolder = 'images/test_cmu';% train, validation, test
if ~exist(outfolder,'dir')
    mkdir([outfolder,'/junctions']);
    mkdir([outfolder,'/non_junctions']);
    mkdir([outfolder,'/gaps']);
    mkdir([outfolder,'/non_gaps']);
end

load(['features/','BSD','_',city,'_',dataset,'.mat']);
filepath = ['images/unclassified/',dataset,'/'];
    
juncs_num = 0;
nonjuncs_num = 0;   
gaps_num = 0;
nongaps_num = 0;

parfor_progress('Generate Dataset', length(routes));
for j=1:length(routes)
    desc = routes(j).BSDs;
    id = routes(j).id;

    img_f = imread([filepath, id, '_front.jpg']);
    img_b = imread([filepath, id, '_back.jpg']);
    img_l = imread([filepath, id, '_left.jpg']);
    img_r = imread([filepath, id, '_right.jpg']);  

    % front
    if desc(1) == 1
        juncs_num = juncs_num + 1;
        imwrite(img_f, fullfile(outfolder,'junctions',sprintf('%s_front.jpg',id)));  
    else
        nonjuncs_num = nonjuncs_num + 1;
        imwrite(img_f, fullfile(outfolder,'non_junctions',sprintf('%s_front.jpg',id)));  
    end

    % back
    if desc(3) == 1
        juncs_num = juncs_num + 1;
        imwrite(img_b, fullfile(outfolder,'junctions',sprintf('%s_back.jpg',id))); 
    else
        nonjuncs_num = nonjuncs_num + 1;
        imwrite(img_b, fullfile(outfolder,'non_junctions',sprintf('%s_back.jpg',id)));
    end 

    % right
    if desc(2) == 1
        gaps_num = gaps_num + 1;
        imwrite(img_r, fullfile(outfolder,'gaps',sprintf('%s_right.jpg',id))); 
    else
        nongaps_num = nongaps_num + 1;
        imwrite(img_r, fullfile(outfolder,'non_gaps',sprintf('%s_right.jpg',id))); 
    end        

    % left
    if desc(4) == 1
        gaps_num = gaps_num + 1;
        imwrite(img_l, fullfile(outfolder,'gaps',sprintf('%s_left.jpg',id))); 
    else
        nongaps_num = nongaps_num + 1;
        imwrite(img_l, fullfile(outfolder,'non_gaps',sprintf('%s_left.jpg',id)));  
    end 
    
    parfor_progress('Generate Dataset');
end