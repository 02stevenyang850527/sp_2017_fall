% Read input texts
fileID = fopen('japan.txt','r');
txt = fscanf(fileID,'%c');
fclose(fileID);
word = unique(txt);


temp = char(zeros(1,71));
temp(1:45)='Japn (es:日本Niorh;fmly.Abutd-k,"S)cELPOCKRgTwj';
temp(46) = word(5);
temp(47:70) = 'v6852H4719%xDG3UIWzFM0B–';
temp(71) = word(1);
word = temp;

char2idx = encoder(word, txt); % Turn char into integer
Y = [char2idx(2:end),10];
X = one_hot(word, char2idx); % Turn integer into one-hot foramt

% Initialization of LSTM
H = 64;
[~,D] = size(word);
Z = H + D; % concat the LSTM state with the input

Wf = randn(Z,H) ./ sqrt(Z/2);
Wi = randn(Z,H) ./ sqrt(Z/2);
Wc = randn(Z,H) ./ sqrt(Z/2);
Wo = randn(Z,H) ./ sqrt(Z/2);
Wy = randn(H,D) ./ sqrt(D/2);

% Wf = zeros(Z,H);
% Wi = zeros(Z,H);
% Wc = zeros(Z,H);
% Wo = zeros(Z,H);
% Wy = zeros(H,D);

bf = zeros(1,H);
bi = zeros(1,H);
bc = zeros(1,H);
bo = zeros(1,H);
by = zeros(1,D);

% parameters
alpha = 1e-3;
time_step = 10;
n_iter = 400000;

solver_mex(X,Y,alpha,time_step,n_iter,Wf,Wi,Wc,Wo,Wy,bf,bi,bc,bo,by);

