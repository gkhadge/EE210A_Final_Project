%Jason Zheng & Gourav Khadge
%EE210A Speech Recognition

[audio_signals, word_labels] = load_audio('audio');
apple = Word('apple');

apple_signals = {};

%plot all the feature vectors for apple signals
for i = 1:15
    apple_signals(i) = {extract_features(audio_signals{i})};
    scatter(apple_signals{i}(1,:), apple_signals{i}(2,:));
    hold on;
end
    
axis([0 4000 0 4000]);
