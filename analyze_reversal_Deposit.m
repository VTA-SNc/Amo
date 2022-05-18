function analyze_reversal_deposit

% animal = {'GCaMP6f_351', 'GCaMP6f_353'}; %GCaMP6f, only puff to water
% animal = {'GrabDA_439','GrabDA_440','GrabDA_445'}; %DA sensor
animal = {'GCaMP7f_459', 'GCaMP7f_460','GCaMP7f_461','GCaMP7f_496','GCaMP7f_497','GCaMP7f_498','GCaMP7f_499'}; %GCaMP7f Reversal (None to water, and puff to water)

commonTr=120;%

M_plot = [];
Beta_linear = [];P_linear = [];Rho = [];Pval = [];Rsq_linear = [];
MSE_exponential = [];
Ci=[];
Trial_learned=[];

odor_response_mean_add=[];
odor_response_late_mean_add=[];
odor_response_addall=[];
odor_response_late_addall=[];

for animal_n = 1:length(animal)
animal{animal_n}
session_type = {'day1','day2','day3'};

plotWin = -2000:7000;
DeltaF_combined = []; DeltaF_combined_red = [];Trial_plot = [];

 for session_n = 1:3
session_type{session_n};
load (strcat('ReversalLearning_',animal{animal_n},'_',session_type{session_n})); %'DeltaF','Trial_number',('DeltaF_tdTom')

Trial_N = cumsum(Trial_number);
mean_freewater = mean(DeltaF((Trial_N(3)+1):Trial_N(4),2001:4000)'); %for '439','440','445'
%mean_freewater = mean(DeltaF((Trial_N(4)+1):Trial_N(5),2001:3000)'); %for
%'459', '460','461','496','497','498','499'
mean_freewater = mean(mean_freewater);

% DeltaF_plot = DeltaF(1:Trial_number(1),:); %reversal from nothing
 DeltaF_plot = DeltaF((Trial_N(1)+1):Trial_N(2),:); %reversal from puff
% DeltaF_plot = DeltaF_tdTom(1:Trial_number(1),:); %tdTom reversal from nothing
%DeltaF_plot = DeltaF_tdTom((Trial_N(1)+1):Trial_N(2),:); %tdTom reversal from puff
DeltaF_combined = [DeltaF_combined;DeltaF_plot];
Trial_plot = [Trial_plot,Trial_number(2)]; % 1:nothing to water 2: puff to water

m_plot = mean(DeltaF_plot);

end

M_plot = [M_plot;m_plot];

Trial_plot2=[0,Trial_plot]
%PSTH for each session
 figure
for bin_n = 1:3%
subplot(3,1,bin_n)

m_plot = mean(DeltaF_combined(sum(Trial_plot2(1:bin_n))+1:sum(Trial_plot2(1:bin_n+1)),:));
s_plot = std(DeltaF_combined(sum(Trial_plot2(1:bin_n))+1:sum(Trial_plot2(1:bin_n+1)),:))/sqrt(size(DeltaF_combined(sum(Trial_plot2(1:bin_n))+1:sum(Trial_plot2(1:bin_n+1))-1,:),1));

errorbar_patch(plotWin,m_plot,s_plot,'b');
axis([-2000,7000,-1.2,2.5])
h=gca;
h.XTick = -2000:2000:6000;
h.XTickLabel = {-2:2:6};
 xlabel('time - odor (s)')
 ylabel('zscore')
 title(session_type{bin_n})
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',10)
set(gcf,'color','w')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Heatmap of signal throughout sessions

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/1.5 scrsz(3)/1.5 scrsz(4)/1.5])
%1) bin the data
trialNum = size(DeltaF_combined,1); binSize = 100;
length_x = plotWin(end)-plotWin(1);
binedF = squeeze(mean(reshape(DeltaF_combined(:,1:length_x),trialNum, binSize,[]),2));
imagesc(binedF,[-3 3]); 
colormap yellowblue
xlabel('time - odor (s)');
colorbar
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

ylabel('trials')
box off
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
set(gca,'FontSize',20)
set(gcf,'color','w')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% response time-course
DeltaF_reversal = DeltaF_combined;

water_response = DeltaF_combined(:,5301:6300);
water_response = mean(water_response');
odor_response = DeltaF_combined(:,2001:3000);
odor_response_addall(animal_n,:,:)=odor_response(1:commonTr,:);
odor_response = mean(odor_response');
odor_response_late = DeltaF_combined(:,4001:5000);
odor_response_late_addall(animal_n,:,:)=odor_response_late(1:commonTr,:);
odor_response_late = mean(odor_response_late');
odor_response_middle = DeltaF_combined(:,3001:4000);
odor_response_middle = mean(odor_response_middle');

water_smooth = movmean(water_response,20);
odor_smooth = movmean(odor_response,20);
odor_late_smooth = movmean(odor_response_late,20);
odor_middle_smooth = movmean(odor_response_middle,20);

odor_response_mean = mean(odor_response(2:20));
odor_response_late_mean = mean(odor_response_late(2:20));

odor_response_mean_add=[odor_response_mean_add odor_response_mean];
odor_response_late_mean_add=[odor_response_late_mean_add odor_response_late_mean];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% peak location
%[deltaF_peak,peak_location] = max(DeltaF_reversal(:,2000:5000)');
[deltaF_peak,peak_location] = max(DeltaF_reversal(:,2500:5000)'); %Excluding activation during too early time window
DeltaF_baseline = DeltaF_reversal(:,1000:2000);

DeltaF_baseline_all = reshape(DeltaF_baseline,1,[]);
diff_peak_baseline = deltaF_peak - 2*std(DeltaF_baseline_all');
ind = find(diff_peak_baseline>0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fitting with exponential function

x = ind';     % trial number
%y = 3000-peak_location(ind)'; % peak from odor onset
y = (2500-peak_location(ind))'; % peak from water onset

beta0 = [2100 3000 -0.03];     % initial parameter for fit


errortest=0
try
    
[beta,R,J,CovB,MSE] = nlinfit(x,y,@exponentialf,beta0);
MSE
MSE_exponential = [MSE_exponential;mse]; %mean squared error

catch
    errortest=1
    
end

if errortest==0

ci = nlparci(beta,R,'jacobian',J);
figure
% Plot the fitted curve
X = 1:length(peak_location);
Y = exponentialf(beta,X);
plot(X,Y,'b'); hold on
hold on

% Plot the original data points
ms = 10;
plot(x,y,'ob','markersize',ms,'markerfacecolor','w')

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
if isempty(trial_learned)||trial_learned==1
    trial_learned = length(Y);
end

Trial_learned = [Trial_learned trial_learned];


elseif errortest==1
 trial_learned = trialNum 
 Trial_learned=[Trial_learned trial_learned];


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % fitting with linear

peak_ind = find(ind<trial_learned); %use only learning phase
peak_trial = ind(peak_ind);

x = peak_trial';     % trial number
%y = (3000-peak_location(ind))'; % peak from water onset
y = (2500-peak_location(peak_trial))'; % peak from water onset

[beta,dev,stats] = glmfit(x,y);

figure
% Plot the fitted curve
yfit = glmval(beta,x,'identity');
plot(x,yfit,'b-'); hold on

% Plot the original data points
ms = 10;
plot(x,y,'ob','markersize',ms,'markerfacecolor','w')

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
p_linear = stats.p
Beta_linear = [Beta_linear;beta_linear'];
P_linear = [P_linear;p_linear'];

mdl = fitlm(x,y);
rsq = mdl.Rsquared.Ordinary;
Rsq_linear = [Rsq_linear;rsq];
ci=coefCI(mdl);
ci=(ci(2,2)-ci(2,1))/2;
%monotonic relation can be tested with Pearson's correlation or Spearman
%rank-order correlation
Ci=[Ci; ci]

[rho, pval] = corr(x,y,'Type','Pearson') %default is Pearson, choose Spearman, or Kendall

Rho = [Rho;rho]
Pval = [Pval;pval]

 end


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% verification of gradual shift
% pick 3 trials and check monotonic relation of peaks
temp1 = find(diff_peak_baseline>0);
temp2 = peak_location(temp1);
temp3 = find(temp1<=trial_learned/3);

bootN = 100;
monotonic_fraction = NaN(1,bootN);
monotonic_fraction_control = NaN(1,bootN);
for i = 1:500

   trial12 = []
    for j=1:bootN;
    temp = randperm(length(temp3),2)';
    trial12 = [trial12 temp];
    end

trial1 = min(trial12,[],1);
trial2 = max(trial12,[],1);
 if size(peak_location)==trial_learned
  trial3 = randi([length(temp1)*2/3,length(temp1)],1,bootN);
 else
  trial3 = randi([length(peak_trial),length(temp2)],1,bootN);
 end
peak_trial1 = (2500-temp2(trial1));
peak_trial2 = (2500-temp2(trial2));
peak_trial3 = (2500-temp2(trial3));

diff_trial12 = peak_trial2 - peak_trial1;
diff_trial23 = peak_trial3 - peak_trial2;

ind = (diff_trial12>0 & diff_trial23>0);
monotonic_fraction(i) = sum(ind)/bootN;


%control

peak_trial1_control = 2500 - randi(2500,1,bootN);

diff_trial12_control = peak_trial2 - peak_trial1_control;

ind = (diff_trial12_control>0 & diff_trial23>0);
monotonic_fraction_control(i) = sum(ind)/bootN;

end

[h,p] = ttest2(monotonic_fraction,monotonic_fraction_control); 

%Histogram for probability of monotonic relation in data and control
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
center_mass_100 = mean(center_mass(2:20));  %mean trial 2-20
center_mass_LP = mean(center_mass(2:(trial_learned/3)));  %1/3 learning phase



%control
Center_mass_100_control = NaN(1,500);
Center_mass_LP_control = NaN(1,500);

for i = 1:500
time_bin_control = randperm(3000);
center_mass_control = time_bin_control.*(delta./sum_delta);
center_mass_control = (3000 - sum(center_mass_control,2))/1000;
center_mass_100_control = mean(center_mass_control(2:20));  %mean trial 2-100
Center_mass_100_control(i) = center_mass_100_control;
center_mass_LP_control = mean(center_mass_control(2:(trial_learned/3)));  %mean trial 2-100
Center_mass_LP_control(i) = center_mass_LP_control;
end

p_center = sum(Center_mass_100_control<=center_mass_100)/500;
p_center2 = sum(Center_mass_LP_control<=center_mass_LP)/500;
p_center_test=num2str(p_center)
p_center2_test=num2str(p_center2)

%Plot center of mass location
figure
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
%line([20,20],[0,3],'Color','[0.7 0.7 0.7]');
ylim([min(center_mass)-0.1,(max(center_mass)+0.1)])

 
%% average of multiple animals

%Plot coefficeint beta for all animal
figure
 boxplot(Beta_linear(:,2))
 hold on
plot(1,Beta_linear(:,2),'ko','markersize',10,'markerfacecolor','w')
hold on
ind = find(P_linear(:,2)<0.05);
if length(ind)>0
plot(1,Beta_linear(ind,2),'ro','markersize',10,'markerfacecolor','w')
end
h=gca;
h.XTick = 0:1000:3000;
h.XTickLabel = {0:1:3};
box off
title('peak')
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('coefficient beta')
set(gca,'FontSize',20)
set(gcf,'color','w')

[h,p_beta]=ttest(Beta_linear(:,2))


%Plot coefficeint beta with confidence interval for all animal
figure
errorbar(1:length(Beta_linear(:,2)),Beta_linear(:,2),Ci,'ko','markersize',10,'markerfacecolor','w');hold on
ind = find(P_linear(:,2)<0.05);
if length(ind)>0
plot(ind,Beta_linear(ind,2),'ro','markersize',10,'markerfacecolor','w')
end
h=gca;
h.XTick = 0:1000:3000;
h.XTickLabel = {0:1:3};
box off
title('peak')
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('coefficient beta')
set(gca,'FontSize',20)
set(gcf,'color','w')


%Plot correlation coefficeint for all animal
figure
boxplot(Rho)
hold on
plot(1,Rho,'ko','markersize',10,'markerfacecolor','w')
hold on
ind = find(Pval<0.05);
if length(ind)>0
plot(1,Rho(ind),'ro','markersize',10,'markerfacecolor','w')
end
box off
title('peak')
set(gca,'tickdir','out')
set(gca,'TickLength',2*(get(gca,'TickLength')))
ylabel('correlation coefficient')
set(gca,'FontSize',20)
set(gcf,'color','w')

[h,p_rho]=ttest(Rho)

Rho

%%
%save([animal{animal_n} ],'monotonic_fraction','monotonic_fraction_control','center_mass','Center_mass_100_control','center_mass_100','Center_mass_LP_control','center_mass_LP','Peak_Sel','peak_location','peak_sel_ind','p_center','p_center2','trial_learned')

end

