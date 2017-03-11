
bw_iters = 1000; %adjustable parameter

[audio_signals, word_labels] = load_audio('audio');
apple = Word('apple');
banana = Word('banana');
kiwi = Word('kiwi');
lime = Word('lime');
orange = Word('orange');
peach = Word('peach');
pineapple = Word('pineapple');


apple_signals = {};

%plot all the feature vectors for apple signals (feature vector length 2)
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



apple.initialize(apple_signals{1}); % use one observation set to iniialize our HMM
[mu_convergence,Sigma_convergence,A_convergence] = apple.trainAll(apple_signals, bw_iters); % train our HMM using the baum welch with 15 iterations

% numiters = apple.trainAll2convergence(apple_signals); % train our HMM using the baum welch with 15 iterations
% numiters
%%
figure(1)
subplot(3,1,1)
semilogy(mu_convergence)
grid on
title('Mu Convergence')
subplot(3,1,2)
semilogy(Sigma_convergence)
grid on
title('Sigma Convergence')
subplot(3,1,3)
semilogy(A_convergence)
grid on
title('A Convergence')