filename = 'rgb1/f1.csv';
M = csvread(filename);

plot(1:1:length(M), M,'LineWidth',3)
hold on

filename = 'rgb1/iou.csv';
M = csvread(filename);

plot(1:1:length(M), M,'LineWidth',3)

filename = 'rgb1/macc.csv';
M = csvread(filename);

plot(1:1:length(M), M,'LineWidth',3)

filename = 'rgb1/oacc.csv';
M = csvread(filename);

plot(1:1:length(M), M,'LineWidth',3)

legend({'F1Score','IoU', 'Mean Accuracy', 'Overall Accuracy'},'FontSize',30,'Location','Southeast');
ylim([0 1])
ylabel('Metrics','FontSize', 30)
xlabel('Epoch','FontSize', 30)
set(gca,'fontsize', 30)