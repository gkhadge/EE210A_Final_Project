% This script generates plots to visualize the Hidden Markov Model
% with the 15 utterances of "apple"

[audio_signals, word_labels] = load_audio('audio');
apple = Word('apple');

% Configuration Settings
apple.setN(15)
apple.setStartAtFirstState(true)
apple.setMarkovLinearTopology(true)

for i = 1:15
    apple_signals(i) = {extract_features(audio_signals{i})};
    banana_signals(i) = {extract_features(audio_signals{i+15})};
    kiwi_signals(i) = {extract_features(audio_signals{i+30})};
    lime_signals(i) = {extract_features(audio_signals{i+45})};
    orange_signals(i) = {extract_features(audio_signals{i+60})};
    peach_signals(i) = {extract_features(audio_signals{i+75})};
    pineapple_signals(i) = {extract_features(audio_signals{i+90})};
%     scatter(apple_signals{i}(1,:), apple_signals{i}(2,:));
%     hold on;
end


% To sort into training data and test data
num_train = 10;
num_expr = 15 - num_train;

% To sort into training data and test data
random_indx = zeros(15,7);
for i = 1:7
    random_indx(:,i)=randperm(15);    
end

apple.initialize(apple_signals{1}); % use one observation set to iniialize our HMM
num_iters = apple.trainAll2convergence(apple_signals(random_indx(1:num_train,1))); % train our HMM using the baum welch with 15 iterations
disp(['Iterations (apple): ', num2str(num_iters)]);
disp('Finished Training')

test_signals = [apple_signals(random_indx(num_train+1:end,1))' 
                banana_signals(random_indx(num_train+1:end,2))'
                kiwi_signals(random_indx(num_train+1:end,3))'
                lime_signals(random_indx(num_train+1:end,4))'
                orange_signals(random_indx(num_train+1:end,5))'
                peach_signals(random_indx(num_train+1:end,6))'
                pineapple_signals(random_indx(num_train+1:end,7))']';

% [error_rate, predicted_labels, confidence] = cross_validate(test_signals, word_labels, word_arr);

%%
close all

self.N = apple.N;
self.A = apple.A;

loop_ind = 0;
% Plot results
for q = random_indx(num_train+1:end,1)'
loop_ind = loop_ind+1;
disp(['Plotting ',num2str(q)])
observation = apple_signals{q};

%function k_o = viterbi_decoder(self, observation)
num_obs = size(observation,2);
log_p = zeros(self.N,num_obs);
psi = zeros(self.N,num_obs);

D = apple.state_likelihood(observation);

log_p(:,1) = log(apple.prior.*D(:,1));

% Forward Pass
for n = 2:num_obs
   for k = 1:self.N 
       arg = log(self.A(:,k)) + log_p(:,n-1);
       [~,ind] = max(arg);
       psi(k,n) = ind;
       log_p(k,n) = log(self.A(psi(k,n),k)) + log_p(psi(k,n),n-1) + log(D(k,n));
   end
end

% Backward Pass
k_o = zeros(num_obs,1);
[~,ind] = max(log_p(:,num_obs));
k_o(num_obs) = ind;

for n = num_obs-1:-1:1
    k_o(n) = psi(k_o(n+1),n+1);
end

figure('position',[100,100,600,500])
subplot(3,1,1)
plot(k_o,'.-')
xlim([1,num_obs])
%yticks(1:apple.N) %matlab 2017 exclusive function
ylim([0.5,apple.N+0.5])
grid on
xlabel('Observations (n)')
ylabel('State')
title('Viterbi Decoded State Transitions')
subplot(3,1,2)
plot(audio_signals{q})
xlim([1,length(audio_signals{q})])
xlabel('Samples')
ylabel('Amplitude')
title('Raw Sound Data')
subplot(3,1,3)
plot(apple_signals{q}(1,:),'ro');
grid on
hold all
plot(apple_signals{q}(2,:),'bo');

obs = 1:num_obs;
for k = 1:apple.N
    plot(obs(k_o==k),apple.mu(1,k)*ones(length(obs(k_o==k)),1),'r.-')
    plot(obs(k_o==k),apple.mu(2,k)*ones(length(obs(k_o==k)),1),'b.-')
end

xlim([1,num_obs])
xlabel('Observations (n), red: 1st spectral feature, blue: 2nd spectral feature')
ylabel('Frequency Amplitude')
title('Spectral Features (Actual and State-Expected)')
% legend('1st','2nd')

topology = 'Linear';
Nstate = 'N15';
Fvector = 'F2';
filename = ['Viterbi_Apple_',topology,'_',Nstate,'_',Fvector,'_',num2str(loop_ind)];
saveas(gcf,filename,'epsc')
end
disp('Transition Matrix:')
disp(apple.A)