close all
states = [1:16,20];
num_states = length(states);
% load('Linear_N3_F2_data.mat');
errors = zeros(num_states,1);
errors_train = zeros(num_states,1);

biases = zeros(num_states,1);
stdevs = zeros(num_states,7);

for i = 1:num_states
    filename = ['Linear_N',num2str(states(i)),'_F2_data.mat'];
    load(filename);
    errors(i) = mean(error_rates);
    errors_train(i) = mean(error_rates_train);
    
    
    groupings = boxplot_groupings_train(:);
    for j = 1:7
        biases(i,j) = mean(overall_confidence_train(groupings==j));
        stdevs(i,j) = std(overall_confidence_train(groupings==j));
    end
end
figure(1)
hold all
plot(states,errors,'rx-')
plot(states,errors_train,'b.-')
legend('Testing Data Error Rate','Training Data Error Rate')
title('Error Rate vs. Numbers of HMM States')
xlabel('Number of HMM States')
ylabel('Average Error Rate')

figure(2)
hold all
plot(states, biases(:,2),'r.-')
plot(states,stdevs(:,2),'b.-')

% topology = 'Linear';
% Nstate = 'N8';
% Fvector = 'F2';
% filename = ['ErrRateVsHMMstates_',topology,'_',Fvector];
% saveas(gcf,filename,'epsc')

%%

%%
close all
featureNs = 1:7;
num_featureNs = length(featureNs);
% load('Linear_N3_F2_data.mat');
errors = zeros(num_featureNs,1);
errors_train = zeros(num_featureNs,1);

for i = 1:num_featureNs
    filename = ['Linear_N10_F',num2str(i),'_data.mat'];
    load(filename);
    errors(i) = mean(error_rates);
    errors_train(i) = mean(error_rates_train);
end
figure(2)
hold all
plot(featureNs,errors,'rx-')
plot(featureNs,errors_train,'b.-')
legend('Testing Data Error Rate','Training Data Error Rate')
title('Error Rate vs. Numbers of Features')
xlabel('Numbers of Features')
ylabel('Average Error Rate')

topology = 'Linear';
Nstate = 'N10';
filename = ['ErrRateVsNumFeatures_',topology,'_',Nstate];
saveas(gcf,filename,'epsc')

%% 

close all
states_Ergodic = [1:6,10];
states_Linear = [1:16,20];
num_states_Ergodic = length(states_Ergodic);
num_states_Linear = length(states_Linear);
% load('Linear_N3_F2_data.mat');
errorsLinear = zeros(num_states_Linear,1);
errorsLinear_train = zeros(num_states_Linear,1);
errorsErgodic = zeros(num_states_Ergodic,1);
errorsErgodic_train = zeros(num_states_Ergodic,1);


for i = 1:num_states_Linear
    filename = ['Linear_N',num2str(states_Linear(i)),'_F2_data.mat'];
    load(filename);
    errorsLinear(i) = mean(error_rates);
    errorsLinear_train(i) = mean(error_rates_train);
    
end


for i = 1:num_states_Ergodic
    
    filename = ['Ergodic_N',num2str(states_Ergodic(i)),'_F2_data.mat'];
    load(filename);
    errorsErgodic(i) = mean(error_rates);
    errorsErgodic_train(i) = mean(error_rates_train);
    
end
figure(1)
hold all
plot(states_Linear,errorsLinear,'rx-')
plot(states_Linear,errorsLinear_train,'r.-')
plot(states_Ergodic,errorsErgodic,'bx-')
plot(states_Ergodic,errorsErgodic_train,'b.-')
legend('Testing Data Error Rate (Linear)','Training Data Error Rate (Linear)','Testing Data Error Rate (Ergodic)','Training Data Error Rate (Ergodic)')
title('Error Rate with Linear and Ergodic Topologies')
xlabel('Number of HMM States')
ylabel('Average Error Rate')
% 
% topology = 'Linear';
Nstate = 'N10';
Fvector = 'F2';
filename = ['ErrRateVsTopology_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')