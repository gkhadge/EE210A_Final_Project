%% Jason Zheng & Gourav Khadge
% EE210A Speech Recognition
% main.m
% This file runs the HMM training and testing, and outputs performance
% statistics averaged over num_trials trials.
% It is con

% Close open figures
close all
%% PARAMETERS 

% Adjustable parameters for HMM
topology = 'Linear';
% topology = 'Ergodic';
num_states = 8;
num_features = 2;
% Adjustable number of trials to average over
num_trials = 50;

% Flag to save plots and data
save_data = false;

%% TESTING

% Strings for saving files
Nstate = ['N', num2str(num_states)];
Fvector = ['F', num2str(num_features)];

[audio_signals, word_labels_orig] = load_audio('audio');
apple = Word('apple');
banana = Word('banana');
kiwi = Word('kiwi');
lime = Word('lime');
orange = Word('orange');
peach = Word('peach');
pineapple = Word('pineapple');

apple.setN(num_states);
banana.setN(num_states);
kiwi.setN(num_states);
lime.setN(num_states);
orange.setN(num_states);
peach.setN(num_states);
pineapple.setN(num_states);

apple.setLinearModel(strcmp(topology,'Linear'));
banana.setLinearModel(strcmp(topology,'Linear'));
kiwi.setLinearModel(strcmp(topology,'Linear'));
lime.setLinearModel(strcmp(topology,'Linear'));
orange.setLinearModel(strcmp(topology,'Linear'));
peach.setLinearModel(strcmp(topology,'Linear'));
pineapple.setLinearModel(strcmp(topology,'Linear'));


apple_signals = {};

% To sort into training data and test data
num_train = 10;
num_expr = 15 - num_train;
word_labels = word_labels_orig([1:num_expr,...
                           16:15+num_expr,...
                           31:30+num_expr,...
                           46:45+num_expr,...
                           61:60+num_expr,...
                           76:75+num_expr,...
                           91:90+num_expr]);
                       
word_labels_train = word_labels_orig([1:num_train,...
                           16:15+num_train,...
                           31:30+num_train,...
                           46:45+num_train,...
                           61:60+num_train,...
                           76:75+num_train,...
                           91:90+num_train]);

%plot all the feature vectors for apple signals (feature vector length 2)
for i = 1:15
    apple_signals(i) = {extract_features(audio_signals{i},num_features)};
    banana_signals(i) = {extract_features(audio_signals{i+15},num_features)};
    kiwi_signals(i) = {extract_features(audio_signals{i+30},num_features)};
    lime_signals(i) = {extract_features(audio_signals{i+45},num_features)};
    orange_signals(i) = {extract_features(audio_signals{i+60},num_features)};
    peach_signals(i) = {extract_features(audio_signals{i+75},num_features)};
    pineapple_signals(i) = {extract_features(audio_signals{i+90},num_features)};
%     scatter(apple_signals{i}(1,:), apple_signals{i}(2,:));
%     hold on;
end

% Set up data collections structures
confusion_matrices = zeros(7,7,num_trials);
confusion_matrices_train = zeros(7,7,num_trials);
overall_confidence = zeros(num_expr*7,num_trials);
overall_confidence_train = zeros(num_train*7,num_trials);
error_rates = zeros(num_trials,1);
error_rates_train = zeros(num_trials,1);
% overall_confidence = [];
% overall_word_labels = [];

% Trial loop
for trial = 1:num_trials
disp(['Trial ',num2str(trial)])


% Sort into training data and test data
% 15 datasets/word, 7 words
random_indx = zeros(15,7);
for i = 1:7
    random_indx(:,i)=randperm(15);    
end


apple.initialize(apple_signals{random_indx(1)}); % use one observation set to iniialize our HMM
num_iters = apple.trainAll2convergence(apple_signals(random_indx(1:num_train,1))); % train our HMM using the baum welch with 15 iterations
disp(['Iterations (apple): ', num2str(num_iters)]);

banana.initialize(banana_signals{random_indx(1)});
num_iters = banana.trainAll2convergence(banana_signals(random_indx(1:num_train,2)));
disp(['Iterations (banana): ', num2str(num_iters)]);

kiwi.initialize(kiwi_signals{random_indx(1)});
num_iters = kiwi.trainAll2convergence(kiwi_signals(random_indx(1:num_train,3)));
disp(['Iterations (kiwi): ', num2str(num_iters)]);

lime.initialize(lime_signals{random_indx(1)});
num_iters = lime.trainAll2convergence(lime_signals(random_indx(1:num_train,4)));
disp(['Iterations (lime): ', num2str(num_iters)]);

orange.initialize(orange_signals{random_indx(1)});
num_iters = orange.trainAll2convergence(orange_signals(random_indx(1:num_train,5)));
disp(['Iterations (orange): ', num2str(num_iters)]);

peach.initialize(peach_signals{random_indx(1)});
num_iters = peach.trainAll2convergence(peach_signals(random_indx(1:num_train,6)));
disp(['Iterations (peach): ', num2str(num_iters)]);

pineapple.initialize(pineapple_signals{random_indx(1)});
num_iters = pineapple.trainAll2convergence(pineapple_signals(random_indx(1:num_train,7)));
disp(['Iterations (pineapple): ', num2str(num_iters)]);

% Add partitioned signals (testing and training) to vectors
test_signals = [apple_signals(random_indx(num_train+1:end,1))' 
                banana_signals(random_indx(num_train+1:end,2))'
                kiwi_signals(random_indx(num_train+1:end,3))'
                lime_signals(random_indx(num_train+1:end,4))'
                orange_signals(random_indx(num_train+1:end,5))'
                peach_signals(random_indx(num_train+1:end,6))'
                pineapple_signals(random_indx(num_train+1:end,7))']';
            
train_signals = [apple_signals(random_indx(1:num_train,1))' 
                banana_signals(random_indx(1:num_train,2))'
                kiwi_signals(random_indx(1:num_train,3))'
                lime_signals(random_indx(1:num_train,4))'
                orange_signals(random_indx(1:num_train,5))'
                peach_signals(random_indx(1:num_train,6))'
                pineapple_signals(random_indx(1:num_train,7))']';

word_arr = [apple banana kiwi lime orange peach pineapple];

%predict_word(apple_signals{7}, word_arr)

% Predict words and gather performance statistics
% Testing Data
[error_rate, predicted_labels, confidence] = cross_validate(test_signals, word_labels, word_arr);
% Training Data
[error_rate_train, predicted_labels_train, confidence_train] = cross_validate(train_signals, word_labels_train, word_arr);

% Store performance data
error_rates(trial) = error_rate;
error_rates_train(trial) = error_rate_train;

confusion_matrix = confusionmat(word_labels,predicted_labels);
confusion_matrix
confusion_matrices(:,:,trial) = confusion_matrix;


confusion_matrix = confusionmat(word_labels_train,predicted_labels_train);
confusion_matrices_train(:,:,trial) = confusion_matrix;

overall_confidence(:,trial) = confidence;
overall_confidence_train(:,trial) = confidence_train;

end %End Trial loop
%% Make plots of performance and save to files


word_strings = {'apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'};

% disp('Experimental')
average_confusion_matrix = mean(confusion_matrices,3)/num_expr;%average_confusion_matrix/num_trials;
average_confusion_matrix

% disp('Train')
average_confusion_matrix_train = mean(confusion_matrices_train,3)/num_train;%average_confusion_matrix/num_trials;
average_confusion_matrix_train


figure(1)
hold all
boxplot_groupings = zeros(size(overall_confidence));
for i=1:length(word_strings)
%     boxplot_groupings(strcmp(overall_word_labels,word_strings(i))) = i;
    boxplot_groupings(strcmp(word_labels,word_strings(i)),:) = i;
%     plot(i*ones(size(conf_i)),conf_i,'x')
end

boxplot(overall_confidence(:),boxplot_groupings(:),'Labels',word_strings)
grid on
ylabel('Confidence')
title('Confidence of Word Classification (Testing Data)')

if save_data
filename = ['ConfBoxPlot_Testing_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')
end

figure(2)
hold all
boxplot_groupings_train = zeros(size(overall_confidence_train));
for i=1:length(word_strings)
%     boxplot_groupings(strcmp(overall_word_labels,word_strings(i))) = i;
    boxplot_groupings_train(strcmp(word_labels_train,word_strings(i)),:) = i;
%     plot(i*ones(size(conf_i)),conf_i,'x')
end

boxplot(overall_confidence_train(:),boxplot_groupings_train(:),'Labels',word_strings)
grid on
ylabel('Confidence')
title('Confidence of Word Classification (Training Data)')

if save_data
filename = ['ConfBoxPlot_Training_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')
end

figure(3)
histogram(error_rates)
xlabel('Error Rate')
ylabel('Instances')
title('Error Rate Histogram (Testing Data)')
filename = ['ErrRateHist_Testing_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')

figure(4)
histogram(error_rates_train)
xlabel('Error Rate')
ylabel('Instances')
title('Error Rate Histogram (Training Data)')

if save_data
filename = ['ErrRateHist_Training_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')
end

disp(['Mean Error Rate: ',num2str(mean(error_rates))])

figure(5)
imshow(average_confusion_matrix, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels

colormap(gca, hot) % # to change the default grayscale colormap 
% colormap(gca, jet) % # to change the default grayscale colormap 
colorbar
axis on
set(gca, 'XTickLabel',word_strings, 'XTick',1:numel(word_strings))
set(gca, 'YTickLabel',word_strings, 'YTick',1:numel(word_strings))
xlabel('Predicted Label')
ylabel('True Label')
title('Confusion Matrix: Word Identification (Testing Data)')

if save_data
filename = ['AvConfMatrix_Testing_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')
end

figure(6)
imshow(average_confusion_matrix_train, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels

colormap(gca, hot) % # to change the default grayscale colormap
% colormap(gca, jet) % # to change the default grayscale colormap 
colorbar
axis on
set(gca, 'XTickLabel',word_strings, 'XTick',1:numel(word_strings))
set(gca, 'YTickLabel',word_strings, 'YTick',1:numel(word_strings))
xlabel('Predicted Label')
ylabel('True Label')
title('Confusion Matrix: Word Identification (Training Data)')

if save_data
filename = ['AvConfMatrix_Training_',topology,'_',Nstate,'_',Fvector];
saveas(gcf,filename,'epsc')
end
%% Save data to mat file
if save_data
filename = [topology,'_',Nstate,'_',Fvector,'_data.mat'];%'Linear_N6_F2_data.mat'];
save(filename,'average_confusion_matrix',...
                'average_confusion_matrix_train',...
                'confusion_matrices',...
                'error_rates',...
                'error_rates_train',...
                'overall_confidence_train',...
                'boxplot_groupings_train',...
                'overall_confidence',...
                'boxplot_groupings',...
                'word_labels')
end