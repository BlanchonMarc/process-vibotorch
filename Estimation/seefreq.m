%close all; clear all;
subplot(4,1,1)
f = 116;
fs = 1000;  
t = 0.1;    
n = [0:1/fs:t];
cycles = t*f;
x = ones(1,length(n));
duty = 50;
oc_samp = fs/f; 
on_samp = (oc_samp * duty)/100;
off_samp = oc_samp - on_samp;
temp = 0;
for i = 1 : ceil(cycles);
    x(temp+on_samp+1:i*oc_samp) = 0;
    temp = temp + oc_samp;
end
plot(n,x(1:length(n)),'LineWidth',2);ylim([-1 1.5]);
title('PolarCam')
set(gca, 'fontsize', 30)


subplot(4,1,2)
f = 50;
fs = 1000;  
t = 0.1;    
n = [0:1/fs:t];
cycles = t*f;
x = ones(1,length(n));
duty = 50;
oc_samp = fs/f; 
on_samp = (oc_samp * duty)/100;
off_samp = oc_samp - on_samp;
temp = 0;
for i = 1 : ceil(cycles);
    x(temp+on_samp+1:i*oc_samp) = 0;
    temp = temp + oc_samp;
end
plot(n,x(1:length(n)),'r','LineWidth',2);ylim([-1 1.5]);
title('NIR')
set(gca, 'fontsize', 30)

subplot(4,1,3)
f = 15;
fs = 1000;  
t = 0.1;    
n = [0:1/fs:t];
cycles = t*f;
x = ones(1,length(n));
duty = 50;
oc_samp = fs/f; 
on_samp = (oc_samp * duty)/100;
off_samp = oc_samp - on_samp;
temp = 0;
for i = 1 : ceil(cycles);
    x(temp+on_samp+1:i*oc_samp) = 0;
    temp = temp + oc_samp;
end
plot(n,x(1:length(n)),'Color', [1.0, .5, 0.0],'LineWidth',2);ylim([-1 1.5]);
title('UCam')
set(gca, 'fontsize', 30)

subplot(4,1,4)
f = 30;
fs = 1000;  
t = 0.1;    
n = [0:1/fs:t];
cycles = t*f;
x = ones(1,length(n));
duty = 50;
oc_samp = fs/f; 
on_samp = (oc_samp * duty)/100;
off_samp = oc_samp - on_samp;
temp = 0;
for i = 1 : ceil(cycles);
    x(temp+on_samp+1:i*oc_samp) = 0;
    temp = temp + oc_samp;
end
plot(n,x(1:length(n)),'k','LineWidth',2);ylim([-1 1.5]);
title('Kinect')
set(gca, 'fontsize', 30)

