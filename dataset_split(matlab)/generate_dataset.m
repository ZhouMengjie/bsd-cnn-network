% generate dataset
clear all
addpath(genpath(pwd));
% addpath('features');
% addpath('images');
city = 'manhattan';
dataset = 'wallstreet5k'; 
outfolder = 'images/test_ws';% train, validation, test
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
        pth = fullfile(outfolder,'junctions',sprintf('%s_front.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_f, pth);  
        end
    else
        nonjuncs_num = nonjuncs_num + 1;
        pth = fullfile(outfolder,'non_junctions',sprintf('%s_front.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_f, pth);  
        end
    end

    % back
    if desc(3) == 1
        juncs_num = juncs_num + 1;
        pth = fullfile(outfolder,'junctions',sprintf('%s_back.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_b, pth); 
        end
    else
        nonjuncs_num = nonjuncs_num + 1;
        pth = fullfile(outfolder,'non_junctions',sprintf('%s_back.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_b, pth);  
        end
    end 

    % right
    if desc(2) == 1
        gaps_num = gaps_num + 1;
        pth = fullfile(outfolder,'gaps',sprintf('%s_right.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_r, pth);  
        end
    else
        nongaps_num = nongaps_num + 1;
        pth = fullfile(outfolder,'non_gaps',sprintf('%s_right.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_r, pth);  
        end
    end        

    % left
    if desc(4) == 1
        gaps_num = gaps_num + 1;
        pth = fullfile(outfolder,'gaps',sprintf('%s_left.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_l, pth); 
        end
    else
        nongaps_num = nongaps_num + 1;
        pth = fullfile(outfolder,'non_gaps',sprintf('%s_left.jpg',id));
        if ~exist(pth,'file')
            imwrite(img_l, pth);  
        end
    end 
    
    parfor_progress('Generate Dataset');
end