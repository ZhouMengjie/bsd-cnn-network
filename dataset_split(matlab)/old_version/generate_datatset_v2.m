% generate dataset for the testing 
clear all
addpath('features');
addpath('images');
city = 'london';
load(['features/','BSD','/','BSD','_',city,'_10_19','.mat']);
filepath = ['images/',city,'_10_19', '/', 'snaps/'];
outfolder = 'images/test';

for i=1:length(routes)
    desc = routes(i).BSDs;
    id = routes(i).id;
    img_f = imread([filepath, id, '_front.jpg']);
    img_b = imread([filepath, id, '_back.jpg']);
    img_l = imread([filepath, id, '_left.jpg']);
    img_r = imread([filepath, id, '_right.jpg']); 
    
    if desc(1) == 1
        imwrite(img_f, fullfile(outfolder,'junctions',sprintf('%s_front.jpg',id))); 
    else
        imwrite(img_f, fullfile(outfolder,'non_junctions',sprintf('%s_front.jpg',id))); 
    end
    
    if desc(3) == 1
        imwrite(img_b, fullfile(outfolder,'junctions',sprintf('%s_back.jpg',id))); 
    else
        imwrite(img_b, fullfile(outfolder,'non_junctions',sprintf('%s_back.jpg',id)));
    end
    
    if desc(2) == 1
        imwrite(img_r, fullfile(outfolder,'gaps',sprintf('%s_right.jpg',id)));
    else
        imwrite(img_r, fullfile(outfolder,'non_gaps',sprintf('%s_right.jpg',id)));
    end
    
    if desc(4) == 1
        imwrite(img_l, fullfile(outfolder,'gaps',sprintf('%s_left.jpg',id)));
    else
        imwrite(img_l, fullfile(outfolder,'non_gaps',sprintf('%s_left.jpg',id)));
    end    
end
