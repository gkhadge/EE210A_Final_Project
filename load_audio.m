%load all audio signals in folder into signals cell and all the labels into
%labels cell
function [audio_signals, word_labels] = load_audio(audio_folder)
    audio_signals = {};
    word_labels = {};

    for word_folder = struct2cell(dir(audio_folder))
        for word_file = struct2cell(dir(sprintf('%s/%s/*.wav', audio_folder, char(word_folder(1)))))
            file_path = sprintf('%s/%s/%s', audio_folder, char(word_folder(1)), char(word_file(1)));
            
            audio_signals(end + 1) = {audioread(file_path)}; 
            word_labels(end + 1) = word_folder(1); 
    end
end

