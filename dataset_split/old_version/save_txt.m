% output .txt file
% read filepath
clear all
outfolder = ['test/','junctions'];
outfolder_n = ['test/','non_junctions'];
files = dir([outfolder '/*.jpg']);
files = {files.name}';

files_n = dir([outfolder_n '/*.jpg']);
files_n = {files_n.name}';

fp = fopen('test.txt','w');
for i=1:length(files)
    filepath = [outfolder, '/',files{i}];
    fprintf(fp,['%s','\n'],filepath);   
end

for i=1:length(files_n)
    filepath = [outfolder_n, '/',files_n{i}];
    fprintf(fp,['%s','\n'],filepath);   
end

fclose(fp);
