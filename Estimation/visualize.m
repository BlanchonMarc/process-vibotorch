GT = fullfile('Pola1/gt/', '*.npy');
PRED = fullfile('Pola1/pred/', '*.npy');
gtnpy = dir(GT);
prednpy = dir(PRED);
temp = 0;
cnt = 0;
MyColorMap = [0, 0, 0;
    0, 0, 255;
    0, 255, 255;
    255, 255, 255;
    0, 255, 0;
    255, 0, 0;
    255, 255, 0;
    125, 125, 125;
    125, 125, 0;
    0, 125, 125];

prop = 0;




colormap lines


mymap = [10 10 10;
    235 217 167
    58 115 138;
    255 118 109;
    137 173 120;
    199 97 96;
    127 127 127;
    255 255 255
    ]./ 255;

mymap2 = [10 10 10;
    184 217 179;
    108 178 213;
    236 233 165;
    235 199 138;
    222 120 110;
    127 127 127;
    255 255 255
    ]./ 255;

for k = 1:length(gtnpy)
    true = 0;
    false = 0;
    GTbaseFileName = gtnpy(k).name;
    GTfullFileName = fullfile('Pola1/gt/', GTbaseFileName);

    PREDbaseFileName = prednpy(k).name;
    PREDfullFileName = fullfile('Pola1/pred/', PREDbaseFileName);


    GTim = readNPY(GTfullFileName);
    PREDim = readNPY(PREDfullFileName);
    
    imgt = squeeze(GTim);
    impred = squeeze(PREDim);
    
    imwrite(uint8(imgt), mymap2, sprintf('Pola1/gtcol/%d.png',k));
    imwrite(uint8(impred), mymap2, sprintf('Pola1/predcol/%d.png',k));
    
%     subplot(2,1,1);
%     imagesc(squeeze(GTim))
%     title('Ground Truth') 
%     axis equal
%     caxis([0 10])
%     axis([0 254 0 inf])
%     subplot(2,1,2); 
%     imagesc(squeeze(PREDim))
%     title('Prediction') 
%     
%     axis equal
%     caxis([0 10]) 
%     axis([0 254 0 inf])
    
%     for i = 1 : size(squeeze(PREDim),1)
%         for j = 1 : size(squeeze(PREDim),2)
%             a = squeeze(PREDim);
%             b = squeeze(GTim);
%             if(a(i,j) == b(i,j))
%                 true = true + 1;
%             else
%                 false = false + 1;
%             end
%         end
%     end
%     prop = prop + (true / (true + false));
    
    [TP, FP, TN, FN] = calError(squeeze(GTim), squeeze(PREDim));
    
    prec = TP / (TP + FP);
    
    rec = TP / (TP + FN);
    if (isnan((2 * prec * rec / (prec + rec)))) 
    else
    temp = temp + (2 * prec * rec / (prec + rec));
    
    end
    cnt = cnt + 1;
    %pause;
end

temp = temp / cnt
%prop = prop / cnt




% mymap = [0.2422    0.1504    0.6603;
% 0.2751    0.2342    0.8710;
% 0.2760    0.3667    0.9829;
% 0.1834    0.5074    0.9798;
% 0.1380    0.6276    0.8973;
% 0.0036    0.7203    0.7917;
% 0.1938    0.7758    0.6251;
% 0.4456    0.8024    0.3909]