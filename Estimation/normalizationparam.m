folder = '/Users/marc/Desktop/rbg/image/output/rgbData/train/';

filePattern = fullfile(folder, '*.png');
f=dir(filePattern);
files={f.name};

for k=1:numel(files)
    fullFileName = fullfile(folder, files{k});

    Im{k} = imread(fullFileName);

end

for i = 1:numel(files)
    Rmean(i) = mean(mean(Im{i}(:,:,1)));
    Gmean(i) = mean(mean(Im{i}(:,:,2)));
    Bmean(i) = mean(mean(Im{i}(:,:,3)));

    Rstd(i) = std2(Im{i}(:,:,1));
    Gstd(i) = std2(Im{i}(:,:,2));
    Bstd(i) = std2(Im{i}(:,:,3));
end


RGBMean = [mean(Rmean) mean(Gmean) mean(Bmean)] / 255
RGBSTD = [mean(Rstd) mean(Gstd) mean(Bstd)] / 255