function analyze_initial_Deposit

animal = {'437', '438','439','440','444','445','446'};
Beta_linear=[];P_linear=[];Trial_learned=[];Beta_onset_linear=[];P_onset_linear=[];
Div1=2;
% for animal_n = 1:length(animal)
% for animal_n = 1:7
for animal_n = 1:7
animal{animal_n}
session_type = {'day1','day2','day3','day4','day5','day6','day7','day8','day9','day10'};
plotWin = -2000:7000;
DeltaF_combined = [];Trial_plot = [];

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])

for session_n = 1:10
session_type{session_n};

load (strcat('FirstTimeLearning_',animal{animal_n},'_',session_type{session_n})); %'Trial_number_lick','DeltaF_licktrial'

%clear Trial_number

Trial_number = Trial_number_lick; %for initial learning
cum_Trial_number = cumsum(Trial_number);
DeltaF = DeltaF_licktrial;%for initial learning
%veDeltaF_zscore = DeltaF_licktrial;%for initial learning

DeltaF_zscore = reshape(DeltaF,1,[]);
DeltaF_zscore = zscore(DeltaF_zscore);
DeltaF_zscore = reshape(DeltaF_zscore,size(DeltaF));

DeltaF_plot = DeltaF_zscore(1:Trial_number(1),:); % 100%cued water trial(First-time learning)
% DeltaF_plot = DeltaF_zscore((cum_Trial_number(3)+1):end,:); %free water 
DeltaF_combined = [DeltaF_combined;DeltaF_plot];
Trial_plot = [Trial_plot,Trial_number(1)]; %choose trial type 4:free water

subplot(10,1,session_n)
m_plot = mean(DeltaF_plot);
s_plot = std(DeltaF_plot)/sqrt(Trial_number(1)); %choose trial type
errorbar_patch(plotWin,m_plot,s_plot,'b');
axis([-1000,5000,-1,5])
h=gca;
h.XTick = -2000:2000:6000;
h.XTickLabel = {-2:2:6};
% xlabel('time - odor (s)')
% ylabel('normalized response')
% title(session_type{session_n})
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',15)
set(gcf,'color','w')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% raster plot green

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])
%1) bin the data
trialNum = size(DeltaF_combined,1); binSize = 100;
length_x = plotWin(end)-plotWin(1);
binedF = squeeze(mean(reshape(DeltaF_combined(:,1:length_x),trialNum, binSize,[]),2));
imagesc(binedF,[-5 5]); %imagesc(binedF,[-0.2 0.2]);
colormap yellowblue
xlabel('time - odor (s)');
h=gca;
h.XTick = [0:10:(length_x/binSize)];
h.XTickLabel = {(plotWin(1)/1000):(plotWin(end)/1000)};
title(animal{animal_n})
hold on;

x2 = [-plotWin(1)/binSize -plotWin(1)/binSize]; % odor
plot(x2,[0 trialNum+0.5],'r')

% divide trigger
for j = 1:length(Trial_plot)-1  
     plot([0 length_x/binSize],[sum(Trial_plot(1:j))+.5 sum(Trial_plot(1:j))+.5],'m','Linewidth',1)   
end

axis([10, 70, 0, trialNum+0.5])
colorbar
ylabel('trials')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smoothing across trials

DeltaF_reversal = DeltaF_combined;
% DeltaF_reversal_smooth = movmean(DeltaF_reversal,3,1); %smooth
DeltaF_reversal_smooth = DeltaF_reversal; %no smooth

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% response time-course

water_response = DeltaF_combined(:,5301:6300);
water_response = mean(water_response');
odor_response = DeltaF_combined(:,2001:3000);
odor_response = mean(odor_response');
odor_response_late = DeltaF_combined(:,4001:5000);
odor_response_late = mean(odor_response_late');
odor_response_middle = DeltaF_combined(:,3001:4000);
odor_response_middle = mean(odor_response_middle');

water_smooth = movmean(water_response,20);
odor_smooth = movmean(odor_response,20);
odor_late_smooth = movmean(odor_response_late,20);
odor_middle_smooth = movmean(odor_response_middle,20);

figure
subplot(1,2,1)
ms = 5;
plot(water_response,'oc','markersize',ms,'markerfacecolor','w')
hold on
% errorbar(5:10:50,[mean(water_response(1:10)),mean(water_response(11:20)),mean(water_response(21:30)),...
%     mean(water_response(31:40)),mean(water_response(41:50))],...
%     [std(water_response(1:10)),std(water_response(11:20)),std(water_response(21:30)),...
%     std(water_response(31:40)),std(water_response(41:50))]/sqrt(10),'-')
plot(water_smooth,'c-','Linewidth',1)
title('water response')
xlabel('trials')
ylabel('response')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')


subplot(1,2,2)
% figure
plot(odor_response,'ob','markersize',ms,'markerfacecolor','w');
hold on
p1=plot(odor_smooth,'b-','Linewidth',1);
% title('odor')

% figure
plot(odor_response_late,'om','markersize',ms,'markerfacecolor','w');
hold on
p2=plot(odor_late_smooth,'m-','Linewidth',1);
% title('odor late')

% plot(odor_response_middle,'og','markersize',ms,'markerfacecolor','w')
% plot(odor_middle_smooth,'g-','Linewidth',1)

legend([p1,p2],{'odor response early','odor response late'})
xlabel('trials')
ylabel('response')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% peak location

% [deltaF_peak,peak_location] = max(DeltaF_reversal_100bin(:,2000:5000)');
[deltaF_peak,peak_location] = max(DeltaF_reversal_smooth(:,2000:5000)');
%[deltaF_peak,peak_location] = max(DeltaF_reversal_smooth(:,2250:5000)');
% peak_location = peak_location + 250;
DeltaF_baseline = DeltaF_reversal_smooth(:,1:2000);
DeltaF_baseline_all = reshape(DeltaF_baseline,1,[]);
diff_peak_baseline = deltaF_peak - 2*std(DeltaF_baseline_all');
ind = find(diff_peak_baseline>0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fitting with exponential function

x = ind';     % trial number
y = 3000-peak_location(ind)'; % peak from odor onset
beta0 = [2100 3000 -0.03];     % initial parameter for fit
Peak_Sel=3000-peak_location(ind);
peak_sel_ind=ind;
    
[beta,R,J,CovB,MSE] = nlinfit(x,y,@exponentialf,beta0);


figure
% Plot the fitted curve
X = 1:length(peak_location);
Y = exponentialf(beta,X);

plot(X,Y,'b'); hold on

hold on

% Plot the original data points
ms = 10;
plot(x,y,'ob','markersize',ms,'markerfacecolor','w')

% modify axis settings
% axis([0 250 0 3000])
h=gca;
h.YTick = 0:1000:3000;
h.YTickLabel = {0:1:3};
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
title('exponential')
xlabel('trial number')
ylabel('response peak from odor (s)')
set(gca,'FontSize',20)
set(gcf,'color','w')

beta_exponential = beta

trial_learned = find(diff(Y)<1,1,'first') %trial n when shift in exponential fit is <1ms per trial
if isempty(trial_learned)
    trial_learned = length(Y);
end



Trial_learned = [Trial_learned trial_learned];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % fitting with linear

peak_ind = find(ind<trial_learned); %use only learning phase
peak_trial = ind(peak_ind);

x = peak_trial';     % trial number
y = (3000-peak_location(peak_trial))'; % peak from water onset

[beta,dev,stats] = glmfit(x,y);

figure
% Plot the fitted curve
yfit = glmval(beta,x,'identity');
plot(x,yfit,'b-'); hold on

% Plot the original data points
ms = 10;
plot(x,y,'ob','markersize',ms,'markerfacecolor','w')

% modify axis settings
% axis([0 250 0 3000])
h=gca;
h.YTick = 0:1000:3000;
h.YTickLabel = {0:1:3};
box off
title('linear')
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
xlabel('trial number')
ylabel('response peak before water (s)')
set(gca,'FontSize',20)
set(gcf,'color','w')

beta_linear = beta
Beta_linear = [Beta_linear;beta_linear'];

p_linear = stats.p
P_linear = [P_linear;p_linear'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% raster plot green and peak

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])
%1) bin the data
trialNum = size(DeltaF_combined,1); binSize = 100;
length_x = plotWin(end)-plotWin(1);
binedF = squeeze(mean(reshape(DeltaF_combined(:,1:length_x),trialNum, binSize,[]),2));
imagesc(binedF,[-5 5]); %imagesc(binedF,[-0.2 0.2]);
colormap yellowblue
xlabel('time - odor (s)');
h=gca;
h.XTick = [0:10:(length_x/binSize)];
h.XTickLabel = {(plotWin(1)/1000):(plotWin(end)/1000)};
title(animal{animal_n})
hold on;

% mark trigger
% x2 = [(-plotWin(1)+1000)/binSize (-plotWin(1)+1000)/binSize];%water
x2 = [-plotWin(1)/binSize -plotWin(1)/binSize]; % odor
plot(x2,[0 trialNum+0.5],'r')

% divide trigger
for j = 1:length(Trial_plot)-1  
     plot([0 length_x/binSize],[sum(Trial_plot(1:j))+.5 sum(Trial_plot(1:j))+.5],'m','Linewidth',1)   
end

% overlay with peak
plot((peak_location(peak_trial)+2000)/binSize,peak_trial+0.5,'rx','Linewidth',1)

axis([10, 70, 0, peak_trial(end)+0.5])
colorbar
ylabel('trials')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% response onset

onset_location = [];onset_trial = [];
% for trial = 1:size(DeltaF_reversal_100bin,1)
%     ind = find(DeltaF_reversal_100bin(trial,2001:6000) > 2*std(DeltaF_baseline_all'));
for trial = 1:size(DeltaF_reversal_smooth,1)
      ind = find(DeltaF_reversal_smooth(trial,2001:5000) > 2*std(DeltaF_baseline_all'));

if length(ind)>0
    onset_location = [onset_location,ind(1)];
    onset_trial = [onset_trial,trial];
end
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % fitting with linear

onset_ind = find(onset_trial<trial_learned); %use only learning phase of peak
onset_trial = onset_trial(onset_ind);

x = onset_trial';     % trial number
y = (3000-onset_location(onset_ind))'; % peak from water onset

[beta,dev,stats] = glmfit(x,y);

figure
% Plot the fitted curve
yfit = glmval(beta,x,'identity');
plot(x,yfit,'b-'); hold on

% Plot the original data points
ms = 10;
plot(x,y,'ob','markersize',ms,'markerfacecolor','w')

% modify axis settings
% axis([0 250 0 3000])
h=gca;
h.YTick = 0:1000:3000;
h.YTickLabel = {0:1:3};
box off
title('linear')
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
xlabel('trial number')
ylabel('response onset before water (s)')
set(gca,'FontSize',20)
set(gcf,'color','w')

beta_onset_linear = beta
Beta_onset_linear = [Beta_onset_linear;beta_onset_linear'];
p_onset_linear = stats.p
P_onset_linear = [P_onset_linear;p_onset_linear'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% raster plot green and onset

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])
%1) bin the data
trialNum = size(DeltaF_combined,1); binSize = 100;
length_x = plotWin(end)-plotWin(1);
binedF = squeeze(mean(reshape(DeltaF_combined(:,1:length_x),trialNum, binSize,[]),2));
imagesc(binedF,[-5 5]); %imagesc(binedF,[-0.2 0.2]);
colormap yellowblue
xlabel('time - odor (s)');
h=gca;
h.XTick = [0:10:(length_x/binSize)];
h.XTickLabel = {(plotWin(1)/1000):(plotWin(end)/1000)};
title(animal{animal_n})
hold on;

% mark trigger
% x2 = [(-plotWin(1)+1000)/binSize (-plotWin(1)+1000)/binSize];%water
x2 = [-plotWin(1)/binSize -plotWin(1)/binSize]; % odor
plot(x2,[0 trialNum+0.5],'r')

% divide trigger
for j = 1:length(Trial_plot)-1  
     plot([0 length_x/binSize],[sum(Trial_plot(1:j))+.5 sum(Trial_plot(1:j))+.5],'m','Linewidth',1)   
end

% overlay with onset
plot((onset_location(onset_ind)+2000)/binSize,onset_trial+0.5,'rx','Linewidth',1)

axis([10, 70, 0, onset_trial(end)+0.5])
colorbar
ylabel('trials')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')

end

figure
subplot(1,3,2)
boxplot(Beta_linear(:,2))
hold on
plot(1,Beta_linear(:,2),'ko')
ind = find(P_linear(:,2)<0.05);
if length(ind)>0
plot(1,Beta_linear(ind,2),'ro')
end
% h=gca;
% h.XTick = 0:1000:3000;
% h.XTickLabel = {0:1:3};
[h,p_beta,ci_beta,stats_beta]=ttest(Beta_linear(:,2));
p_1_str = num2str(p_beta);
t_1_str =stats_beta.tstat;

box off
title({'peak shift', 'p=' p_1_str ' t=' t_1_str})
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('coefficient beta (ms/trial)')
set(gca,'FontSize',10)
set(gcf,'color','w')



subplot(1,3,3)
boxplot(Beta_onset_linear(:,2))
hold on
plot(1,Beta_onset_linear(:,2),'ko')
ind = find(P_onset_linear(:,2)<0.05);
if length(ind)>0
plot(1,Beta_onset_linear(ind,2),'ro')
end
% h=gca;
% h.XTick = 0:1000:3000;
% h.XTickLabel = {0:1:3};
[h,p_beta_onset,ci_beta_onset,stats_beta_onset]=ttest(Beta_onset_linear(:,2))
p_2_str = num2str(p_beta_onset);
t_2_str =stats_beta_onset.tstat;
box off
title({'onset shift', 'p=' p_2_str ' t=' t_2_str})
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('coefficient beta (ms/trial)')
set(gca,'FontSize',10)
set(gcf,'color','w')




subplot(1,3,1)
boxplot(Trial_learned)
hold on
plot(1,Trial_learned,'ko')
% h=gca;
% h.XTick = 0:1000:3000;
% h.XTickLabel = {0:1:3};
box off
title({'learning phase';' ';' ';' ' ;' '})
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('trials')
set(gca,'FontSize',10)
set(gcf,'color','w')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% verification of gradual shift
% pick 3 trials and check monotonic relation of peaks
% y = (3000-peak_location(peak_trial))'; % peak from water onset
temp1 = find(diff_peak_baseline>0);
temp2 = peak_location(temp1);
temp3 = find(temp1<=trial_learned/Div1);

bootN = 100;
monotonic_fraction = NaN(1,bootN);
monotonic_fraction_control = NaN(1,bootN);
for i = 1:500
%trial12 = randi(round((length(peak_trial))/2),2,bootN); %use first half of learning phase
%trial12 = randi(round(length(find(peak_trial<=(trial_learned/3)))),2,bootN); %use first half of learning phase


   trial12 = []
    for j=1:bootN;
    temp = randperm(length(temp3),2)';
    trial12 = [trial12 temp];
    end

% trial12 = randi(length(peak_trial)-1,2,bootN);
trial1 = min(trial12,[],1);
trial2 = max(trial12,[],1);

 if size(peak_location)==trial_learned
  trial3 = randi([length(temp1)*2/3,length(temp1)],1,bootN);
 else
  trial3 = randi([length(peak_trial),length(temp2)],1,bootN);
 end
peak_trial1 = (3000-temp2(trial1));
peak_trial2 = (3000-temp2(trial2));
peak_trial3 = (3000-temp2(trial3));

diff_trial12 = peak_trial2 - peak_trial1;
diff_trial23 = peak_trial3 - peak_trial2;

ind = (diff_trial12>0 & diff_trial23>0);
monotonic_fraction(i) = sum(ind)/bootN;

% %control1
% % trial1_control = trial12(1,:);
% % trial2_control = trial12(2,:);
% trial1_control = trial2;
% trial2_control = trial1;
% 
% peak_trial1 = (3000-peak_location(trial1_control));
% peak_trial2 = (3000-peak_location(trial2_control));
% peak_trial3 = (3000-peak_location(trial3));
% 
% diff_trial12 = peak_trial2 - peak_trial1;
% diff_trial23 = peak_trial3 - peak_trial2;
% 
% ind = (diff_trial12>0 & diff_trial23>0);
% monotonic_fraction_control(i) = sum(ind)/bootN;

%control2

peak_trial1_control = 3000 - randi(3000,1,bootN);

diff_trial12_control = peak_trial2 - peak_trial1_control;

ind = (diff_trial12_control>0 & diff_trial23>0);
monotonic_fraction_control(i) = sum(ind)/bootN;

end

[h,p] = ttest2(monotonic_fraction,monotonic_fraction_control); 
figure
histogram(monotonic_fraction,'BinWidth',0.02)
hold on
histogram(monotonic_fraction_control,'BinWidth',0.02)
ylabel('fraction')
xlabel('probability of monotonic relation')
legend('data','control')
title(sprintf('p = %1.2e',p))
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% contour

DeltaF_smooth = smoothdata(DeltaF_combined(1:trial_learned,:),'movmean',10);
scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])
%1) bin the data
trialNum = size(DeltaF_smooth,1); binSize = 100;
length_x = plotWin(end)-plotWin(1);
binedF = squeeze(mean(reshape(DeltaF_smooth(:,1:length_x),trialNum, binSize,[]),2));
% imagesc(binedF,[-5 5]); %imagesc(binedF,[-0.2 0.2]);
% contour(flip(binedF),0:0.4:2.5,'LineWidth',3);
contour(flip(binedF),'LineWidth',3);
colormap parula
xlabel('time - odor (s)');
h=gca;
h.XTick = [0:10:(length_x/binSize)];
h.XTickLabel = {(plotWin(1)/1000):(plotWin(end)/1000)};
title(animal{animal_n})
hold on;

x2 = [-plotWin(1)/binSize -plotWin(1)/binSize]; % odor
plot(x2,[0 trialNum+0.5],'r')

axis([10, 70, 0, trialNum+0.5])
colorbar
ylabel('trials')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% center of mass

time_bin = 1:3000;
delta = DeltaF_combined(:,2000:4999);
min_delta = min(delta,[],'all');
if min_delta<0
    delta = delta - min_delta;
end
sum_delta = sum(delta,2);
center_mass = time_bin.*(delta./sum_delta);
center_mass = (3000 - sum(center_mass,2))/1000;
center_mass_100 = mean(center_mass(2:100));  %mean trial 2-100
center_mass_LP = mean(center_mass(2:(trial_learned/3)));  %mean trial 2-100



%control
Center_mass_100_control = NaN(1,500);
Center_mass_LP_control = NaN(1,500);

for i = 1:500
time_bin_control = randperm(3000);
center_mass_control = time_bin_control.*(delta./sum_delta);
center_mass_control = (3000 - sum(center_mass_control,2))/1000;
center_mass_100_control = mean(center_mass_control(2:100));  %mean trial 2-100
Center_mass_100_control(i) = center_mass_100_control;
center_mass_LP_control = mean(center_mass_control(2:(trial_learned/3)));  %mean trial 2-100
Center_mass_LP_control(i) = center_mass_LP_control;
end

p_center = sum(Center_mass_100_control<=center_mass_100)/500;
p_center2 = sum(Center_mass_LP_control<=center_mass_LP)/500;
p_center_test=num2str(p_center)
p_center2_test=num2str(p_center2)

figure
subplot(1,2,1)
plot(center_mass)
title('center of mass')
xlabel('trials')
ylabel('s before water')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')
line([0,length(center_mass)],[1.5,1.5],'Color','[0.7 0.7 0.7]');
line([100,100],[0,3],'Color','[0.7 0.7 0.7]');
ylim([min(center_mass)-0.1,(max(center_mass)+0.1)])

subplot(1,2,2)
histogram(Center_mass_100_control)
hold on
plot([center_mass_LP center_mass_LP], [0 100],'r--')
title([sprintf('100 tr p = %1.2e',p_center); sprintf('  LP/3 p = %1.2e',p_center2)])

xlabel('s before water')
ylabel('histogram')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')
%%
%save([animal{animal_n}],'monotonic_fraction','monotonic_fraction_control','center_mass','Center_mass_100_control','center_mass_100','Center_mass_LP_control','center_mass_LP','Peak_Sel','peak_location','peak_sel_ind','p_center','p_center2','Beta_linear','Beta_onset_linear','Trial_learned')
