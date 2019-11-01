% generate dataset
clear all
addpath('features');
addpath('images');
city_list = {'bath';'bristol';'cambridge';'cheltenham';'coventry';'derby';'glasgow';...
    'leeds';'liverpool';'livingston';'manchester';'newcastle';'norwich';'sheffield';...
    'southampton';'plymouth';'preston';'wakefield';'walsall';'wolverhampton';'york';...
    'nottingham';'leicester';'cardiff';'belfast';'brighton';'aberdeen';'inverness';...
    'durham';'birmingham';'dublin';'lyon';'helsinki';'berlin';'amsterdam';'madrid';...
    'vienna';'athens';'prague';'milan';'miami';'dallas';'atlanta';'chicago';'columbus';...
    'calgary';'edmonton';'ottawa';'montreal';'vancouver'};

load('dataset_split_v2','set_split');
outfolder_tr = 'images/train';
outfolder_te = 'images/test';

for i=1:length(city_list)
    city = city_list{i,1};
    junctions_train = set_split(i).junctions_train;
    nonjunctions_train = set_split(i).nonjunctions_train;
    junctions_test = set_split(i).junctions_test;
    nonjunctions_test = set_split(i).nonjunctions_test;
    
    gaps_train = set_split(i).gaps_train;
    nongaps_train = set_split(i).nongaps_train;
    gaps_test = set_split(i).gaps_test;
    nongaps_test = set_split(i).nongaps_test;
    
    juncs_train_num = 0;
    nonjuncs_train_num = 0;
    juncs_test = 0;
    nonjuncs_test = 0;
    
    gaps_train_num = 0;
    nongaps_train_num = 0;
    gaps_test_num = 0;
    nongaps_test_num = 0;
    
    load(['features/','BSD','/','BSD','_',city,'_10_19','.mat']);
    %filepath = ['images/Images/','BSD','_',city,'_10_19', '/', 'snaps/'];
    filepath = 'images/snaps/';
    for j=1:length(routes)
        desc = routes(j).BSDs;
        id = routes(j).id;
        img_f = imread([filepath, id, '_front.jpg']);
        img_b = imread([filepath, id, '_back.jpg']);
        img_l = imread([filepath, id, '_left.jpg']);
        img_r = imread([filepath, id, '_right.jpg']);  
        
        % front
        if desc(1) == 1
            juncs_train_num = juncs_train_num + 1;
            if juncs_train_num <= junctions_train
                %imwrite(img_f, fullfile(outfolder_tr,'junctions',sprintf('%d.jpg',juncs_train_num)));
                imwrite(img_f, fullfile(outfolder_tr,'junctions',sprintf('%s_front.jpg',id)));            
            else
                if juncs_train_num <= junctions_train + junctions_test
                    %imwrite(img_f, fullfile(outfolder_te,'junctions',sprintf('%d.jpg',juncs_train_num -junctions_train)));
                    imwrite(img_f, fullfile(outfolder_te,'junctions',sprintf('%s_front.jpg',id)));
                end
            end            
        else
            nonjuncs_train_num = nonjuncs_train_num + 1;
            if nonjuncs_train_num <= nonjunctions_train
                %imwrite(img_f, fullfile(outfolder_tr,'non_junctions',sprintf('%d.jpg',nonjuncs_train_num)));
                imwrite(img_f, fullfile(outfolder_tr,'non_junctions',sprintf('%s_front.jpg',id)));
            else
                if nonjuncs_train_num <= nonjunctions_train + nonjunctions_test
                    %imwrite(img_f, fullfile(outfolder_te,'non_junctions',sprintf('%d.jpg',nonjuncs_train_num - nonjunctions_train)));
                    imwrite(img_f, fullfile(outfolder_te,'non_junctions',sprintf('%sfront.jpg',id)));
                end
            end                          
        end
        
        % back
        if desc(3) == 1
            juncs_train_num = juncs_train_num + 1;
            if juncs_train_num <= junctions_train
                %imwrite(img_b, fullfile(outfolder_tr,'junctions',sprintf('%d.jpg',juncs_train_num)));
                imwrite(img_b, fullfile(outfolder_tr,'junctions',sprintf('%s_back.jpg',id)));
            else
                if juncs_train_num <= junctions_train + junctions_test
                    %imwrite(img_b, fullfile(outfolder_te,'junctions',sprintf('%d.jpg',juncs_train_num -junctions_train)));
                    imwrite(img_b, fullfile(outfolder_te,'junctions',sprintf('%s_back.jpg',id)));
                end
            end            
        else
            nonjuncs_train_num = nonjuncs_train_num + 1;
            if nonjuncs_train_num <= nonjunctions_train
                %imwrite(img_b, fullfile(outfolder_tr,'non_junctions',sprintf('%d.jpg',nonjuncs_train_num)));
                imwrite(img_b, fullfile(outfolder_tr,'non_junctions',sprintf('%s_back.jpg',id)));
            else
                if nonjuncs_train_num <= nonjunctions_train + nonjunctions_test
                    %imwrite(img_b, fullfile(outfolder_te,'non_junctions',sprintf('%d.jpg',nonjuncs_train_num - nonjunctions_train)));
                    imwrite(img_b, fullfile(outfolder_te,'non_junctions',sprintf('%s_back.jpg',id)));
                end
            end                          
        end 
        
        % right
        if desc(2) == 1
            gaps_train_num = gaps_train_num + 1;
            if gaps_train_num <= gaps_train
                %imwrite(img_r, fullfile(outfolder_tr,'gaps',sprintf('%d.jpg',gaps_train_num)));
                imwrite(img_r, fullfile(outfolder_tr,'gaps',sprintf('%s_right.jpg',id)));
            else
                if gaps_train_num <= gaps_train + gaps_test
                    %imwrite(img_r, fullfile(outfolder_te,'gaps',sprintf('%d.jpg',gaps_train_num -gaps_train)));
                    imwrite(img_r, fullfile(outfolder_te,'gaps',sprintf('%s_right.jpg',id)));
                end
            end            
        else
            nongaps_train_num = nongaps_train_num + 1;
            if nongaps_train_num <= nongaps_train
                %imwrite(img_r, fullfile(outfolder_tr,'non_gaps',sprintf('%d.jpg',nongaps_train_num)));
                imwrite(img_r, fullfile(outfolder_tr,'non_gaps',sprintf('%s_right.jpg',id)));
            else
                if nongaps_train_num <= nongaps_train + nongaps_test
                    %imwrite(img_r, fullfile(outfolder_te,'non_gaps',sprintf('%d.jpg',nongaps_train_num - nongaps_train)));
                    imwrite(img_r, fullfile(outfolder_te,'non_gaps',sprintf('%s_right.jpg',id)));
                end
            end                          
        end        

        % left
        if desc(4) == 1
            gaps_train_num = gaps_train_num + 1;
            if gaps_train_num <= gaps_train
                %imwrite(img_l, fullfile(outfolder_tr,'gaps',sprintf('%d.jpg',gaps_train_num)));
                imwrite(img_l, fullfile(outfolder_tr,'gaps',sprintf('%s_left.jpg',id)));
            else
                if gaps_train_num <= gaps_train + gaps_test
                    %imwrite(img_l, fullfile(outfolder_te,'gaps',sprintf('%d.jpg',gaps_train_num -gaps_train)));
                    imwrite(img_l, fullfile(outfolder_te,'gaps',sprintf('%s_left.jpg',id)));
                end
            end            
        else
            nongaps_train_num = nongaps_train_num + 1;
            if nongaps_train_num <= nongaps_train
                %imwrite(img_l, fullfile(outfolder_tr,'non_gaps',sprintf('%d.jpg',nongaps_train_num)));
                imwrite(img_l, fullfile(outfolder_tr,'non_gaps',sprintf('%s_left.jpg',id)));
            else
                if nongaps_train_num <= nongaps_train + nongaps_test
                    %imwrite(img_l, fullfile(outfolder_te,'non_gaps',sprintf('%d.jpg',nongaps_train_num - nongaps_train)));
                    imwrite(img_l, fullfile(outfolder_te,'non_gaps',sprintf('%s_left.jpg',id)));
                end
            end                          
        end                 
    end    
end