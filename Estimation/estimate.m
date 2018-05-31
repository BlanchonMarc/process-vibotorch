GT = fullfile('rgb1/gt/', '*.npy');
PRED = fullfile('rgb1/pred/', '*.npy');
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
rate = 0;
b = 0;
hit = 0;
miss = 0;


prob = zeros(1,11);
classpresence = zeros(1,11);

for k = 1:length(gtnpy)

    GTbaseFileName = gtnpy(k).name;
    GTfullFileName = fullfile('rgb1/gt/', GTbaseFileName);

    PREDbaseFileName = prednpy(k).name;
    PREDfullFileName = fullfile('rgb1/pred/', PREDbaseFileName);


    GTim = readNPY(GTfullFileName);
    PREDim = readNPY(PREDfullFileName);
    GTim = squeeze(GTim);
    PREDim = squeeze(PREDim);
    
    [TP, FP, TN, FN] = calError(squeeze(GTim), squeeze(PREDim));
    
    prec = TP / (TP + FP);
    
    rec = TP / (TP + FN);
    if (isnan((2 * prec * rec / (prec + rec)))) 
    else
    temp = temp + (2 * prec * rec / (prec + rec));
    
    end
    cnt = cnt + 1;
    
    for i = 0 : 10
        
        if sum((GTim == i)) ~= 0
            pred = (PREDim == i);
            gt = (GTim == i);
            [rate b hit miss] = hitRates(pred , gt);
            prob(i+1) = prob(i+1) + (hit/(hit+miss));
            classpresence(i+1) = classpresence(i+1) + 1; 
        end
    end

end

f1 = temp / cnt
prob
classpresence
prob ./ classpresence

classes = ["unlabel" , "sky" , "water" , "window", "road", "car", "building", "none"]
c = categorical({ 'unlabel'    ,'sky'   , 'water'   , 'window'   , 'road'    ,'car'    ,'building'    ,'none'})
proba = (prob ./ classpresence) * 100;
graph = proba(1:8)

p1 = bar(c(1),graph(1));
hold on
p2 = bar(c(2),graph(2));
p3 = bar(c(3),graph(3));
p4 = bar(c(4),graph(6));
p5 = bar(c(5),graph(5));
p6 = bar(c(6),graph(6));
p7 = bar(c(7),graph(7));
p8 = bar(c(8),graph(8));

map = [10 10 10;
    184 217 179;
    108 178 213;
    236 233 165;
    235 199 138;
    222 120 110;
    127 127 127;
    255 255 255
    ]./ 255;


set(p1,'FaceColor',map(1,:));
set(p2,'FaceColor',map(2,:));
set(p3,'FaceColor',map(3,:));
set(p4,'FaceColor',map(4,:));
set(p5,'FaceColor',map(5,:));
set(p6,'FaceColor',map(6,:));
set(p7,'FaceColor',map(7,:));
set(p8,'FaceColor',map(8,:));

set(gca,'fontsize',30)
