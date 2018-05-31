folder = '/Users/marc/Documents/Water_shed_Mask/';
Output_folder = '/Users/marc/Documents/Water_shed_Mask/output/';

filePattern = fullfile(folder, '*.png');
f=dir(filePattern);
files={f.name};

for k=1:numel(files)
    fullFileName = fullfile(folder, files{k});

    Im{k} = imread(fullFileName);

end

for i = 1:numel(files)
    Im{i} = rgb2gray(Im{i});
    Im{i}(Im{i}==255) = 0;
    imwrite(Im{i}, [Output_folder files{i}],'png');
end