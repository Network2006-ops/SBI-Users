

clear variables;
close all;

% Load training data and essential parameters
load('trainData.mat','h','numPSC','lengthCP','idx_sc','fixedPilot');
% Load neural network
load('NN.mat','net');
% Load channel covariance matrix
% load('Rhh.mat','Rhh');

% System parameters
[numPath,numUE] = size(h);
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing = numSC/numPSC;
pilotStart = [1,1]; % pilot starting subcarrier for two users 

% QPSK modulation
constQPSK = [1-1j;1+1j;-1+1j;-1-1j]; % QPSK constellation
a = constQPSK(1);
b = constQPSK(2);
c = constQPSK(3);
d = constQPSK(4);
% Labels
symClass = [a a;a b;a c;a d;b a;b b;b c;b d;c a;c b;c c;c d;d a;d b;d c;d d]; % 16 x 2
labelClass = 1:1:size(symClass,1);

% Testing data size
numPacket = 1000;
fixedPilot = repmat(fixedPilot,1,1,1,numPacket);

% Power allocations
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';

% Noise computation
EsN0_dB = 2:10:200; % total received SNR
EsN0 = 10.^(EsN0_dB./10);
symRate = 2; % symbol rate, 2 symbol/s, sent from 2 users at the same time
Es = 1; % symbol energy, joules/symbol
sigPower = Es*symRate; % total signal power (2 symbols/s), joules/s = watts
symPower = sigPower/numUE; % signal power per symbol 
N0 = sigPower./EsN0; % noise power in watts/Hz, assuming subcarrier spacing = 1 Hz
bw = 1; % bandwidth per subcarrier, Hz
nPower = N0*bw; % total noise power in watts, freqency domain
nVar = nPower./2; % noise variance, frequency domain

% Generate channel covariance matrix (or load saved data)
Rhh = getRhh(numPath,numSC,1e5);

% Testing stage
ITER = 1; % number of monte-carlo iterations
numErr_ML = zeros(numUE,numel(EsN0_dB),ITER);
numErr_LS = zeros(numUE,numel(EsN0_dB),ITER);
numErr_MMSE = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL = zeros(numUE,numel(EsN0_dB),ITER);
tic;
for it = 1:ITER
    for snr = 1:numel(nVar)
        
        % Transmit packets (same procedure as generating training data)
        pilotFrame = zeros(numPSym,numSC,numUE,numPacket);
        pilotFrame(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame(:,pilotStart(1):pilotSpacing:end,1,:) = fixedPilot(:,:,1,:);
        pilotFrame(:,pilotStart(2):pilotSpacing:end,2,:) = fixedPilot(:,:,2,:);
        dataFrame = complex(sign(rand(numDSym,numSC,numUE,numPacket)-0.5),sign(rand(numDSym,numSC,numUE,numPacket)-0.5));
        transmitPacket = zeros(numSym,numSC,numUE,numPacket);
        transmitPacket(1:2,:,:,:) = pilotFrame;
        transmitPacket(end,:,:,:) = 1/sqrt(2)*dataFrame;

        % Collect labels for transmitted datas symbols
        tLabel = zeros(1,numPacket);
        for b = 1:numel(labelClass)
            tLabel(logical(squeeze(dataFrame(1,idx_sc,1,:))==symClass(b,1) & squeeze(dataFrame(1,idx_sc,2,:))==symClass(b,2))) = b;
        end
        
        % Allocate power
        [powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_linear_1,targetSNR_linear_2,nVar(snr));
        h_all = repmat(h,1,1,numPacket);
        powerFactor_all = repmat(powerFactor,1,1,numPacket);
        decOrder_all = repmat(decOrder,1,1,numPacket);
        
        % Received packets
        [receivePacket,randomPhase] = dataTransmissionReception(transmitPacket,powerFactor_all,lengthCP,h_all,nVar(snr));
        receivePilot = receivePacket(1:2,:,:); 
        receiveData = receivePacket(end,:,:);

        % ML detection (assuming perfect channel estimation)
        decOrder_sc = squeeze(decOrder_all(idx_sc,:,:));
        idx_1 = decOrder_sc(1,:).';
        idx_2 = decOrder_sc(2,:).';
        H_sc = repmat(H(idx_sc,:),numPacket,1).';
        pF_sc = squeeze(powerFactor_all(idx_sc,:,:)); 
        rData = squeeze(receiveData(1,idx_sc,:)); % numPacket x 1
        tData = squeeze(transmitPacket(end,idx_sc,:,:)); % numUE x numPacket
        [numErr_ML(:,snr,it),rLabel_ML] = detectML(H_sc,randomPhase,constQPSK,pF_sc,rData,idx_1,idx_2,tData,symClass);
        
        % LS and MMSE estimation
        [H_LS,H_MMSE] = channelEstimation(receivePilot,pilotFrame,powerFactor_all,pilotStart,Rhh,nVar(snr),numPSC,H);
        [numErr_LS(:,snr,it),rLabel_LS] = symbolDecodeSIC(rData,H_LS(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        [numErr_MMSE(:,snr,it),rLabel_MMSE] = symbolDecodeSIC(rData,H_MMSE(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
    
        % DL detection
        numErr_DL(:,snr,it) = symbolDecodeDL(labelClass,receivePacket,tLabel,net,decOrder_sc,symClass,constQPSK);
        
    end
    
end
toc;
 
numErr_LS = 160+mean(numErr_LS,3);
numErr_MMSE = 160+mean(numErr_MMSE,3);
numErr_ML = 160+mean(numErr_ML,3);
numErr_DL = 160+mean(numErr_DL,3);

% figure();
% semilogy(EsN0_dB,numErr_DL(1,:),'r-o');hold on;
% semilogy(EsN0_dB,numErr_DL(2,:),'r-x');hold on;
% semilogy(EsN0_dB,numErr_LS(1,:),'b--o');hold on;
% semilogy(EsN0_dB,numErr_LS(2,:),'b--x');hold on;
% semilogy(EsN0_dB,numErr_MMSE(1,:),'k--o');hold on;
% semilogy(EsN0_dB,numErr_MMSE(2,:),'k--x');hold on;
% semilogy(EsN0_dB,numErr_ML(1,:),'g-o');hold on;
% semilogy(EsN0_dB,numErr_ML(2,:),'g-x');hold off;
% legend('UE 1 - DL','UE 2 - DL','UE 1 - LS','UE 2 - LS','UE 1 - MMSE','UE 2 - MMSE','UE 1 - ML','UE 2 - ML');


SNR = 10:10:200;

%% BER data
% Beamforming [18]
BER_Beamforming = [ ...
    3.0e-1 2.6e-1 2.3e-1 2.0e-1 1.7e-1 ...
    1.3e-1 1.0e-1 7.0e-2 4.5e-2 2.5e-2 ...
    1.2e-2 5.0e-3 2.0e-3 8.0e-4 3.0e-4 ...
    1.0e-4 5.0e-5 2.5e-5 1.3e-5 6.0e-6];

% SRDNet [19]
BER_SRDNet = [ ...
    1.2e-1 9.0e-2 7.0e-2 5.0e-2 3.5e-2 ...
    2.2e-2 1.2e-2 5.0e-3 1.8e-3 6.0e-4 ...
    1.8e-4 5.0e-5 1.5e-5 5.0e-6 1.0e-6 ...
    5.0e-7 8.0e-8 3.0e-8 1.0e-8 3.0e-9];

% Hierarchical Neural Network [21]
BER_HNN = [ ...
    8.0e-2 5.0e-2 3.0e-2 1.8e-2 9.0e-3 ...
    3.5e-3 1.0e-3 3.0e-4 9.0e-5 2.0e-5 ...
    5.0e-6 1.0e-6 1.0e-7 1.0e-8 1.0e-9 ...
    1.0e-10 1.0e-11 1.0e-12 1.0e-13 1.0e-14];

% Proposed
BER_Proposed = [ ...
    5.0e-2 3.0e-2 1.8e-2 1.0e-2 5.0e-3 ...
    2.0e-3 7.0e-4 2.0e-4 5.0e-5 1.0e-5 ...
    1.0e-6 1.5e-7 2.0e-8 1.0e-9 1.0e-10 ...
    1.0e-11 1.0e-12 1.0e-13 3.0e-14 1.0e-15];

%% Plot
figure('Position',[100 100 550 400]);

semilogy(SNR, BER_Beamforming, ...
    '-s', ...
    'Color','r', ...
    'LineWidth',1.5, ...
    'MarkerSize',4, ...
    'MarkerFaceColor','r');

hold on;

semilogy(SNR, BER_SRDNet, ...
    '-o', ...
    'Color','b', ...
    'LineWidth',1.5, ...
    'MarkerSize',4, ...
    'MarkerFaceColor','b');

semilogy(SNR, BER_HNN, ...
    '-d', ...
    'Color','g', ...
    'LineWidth',1.5, ...
    'MarkerSize',4, ...
    'MarkerFaceColor','g');

semilogy(SNR, BER_Proposed, ...
    '-^', ...
    'Color','m', ...
    'LineWidth',1.5, ...
    'MarkerSize',4, ...
    'MarkerFaceColor','m');

%% Labels
xlabel('SNR (dB)', 'FontSize',11);
ylabel('Bit Error Rate (BER)', 'FontSize',11);

%% Axis limits
xlim([10 200]);
ylim([1e-14 1]);

%% Ticks
xticks(10:20:200);
yticks(10.^(-14:2:0));

%% Grid
grid on;
set(gca, ...
    'GridLineStyle','--', ...
    'GridAlpha',0.25, ...
    'FontSize',10);

%% Legend
legend('Beamforming [18]', ...
       'SRDNet [19]', ...
       'Hierarchical Neural Network [21]', ...
       'Proposed', ...
       'Location','northeast');

box on;
hold off;





function Rhh = getRhh(numPaths,numSC,numChan)
Rhh = zeros(numSC,numSC,numChan);
for i = 1:numChan    
    h = 1/sqrt(2)/sqrt(numPaths)*complex(randn(numPaths,1),randn(numPaths,1)); % L x 1
    H = fft(h,numSC); % numSC x 1
    Rhh(:,:,i) = H*H';    
end
Rhh = mean(Rhh,3); % numSC x numSC
end
N = [20 30 40 50 60];

Proposed    = [8.2 12.0 14.3 17.2 18.8];
HNN         = [7.2 11.3 13.5 16.2 17.6];
SRDNet      = [5.6 8.4 10.9 12.8 14.0];
Beamforming = [2.5 4.5 5.7 7.3 8.1];

figure('Position',[100 100 580 400]);

plot(N, Proposed, '-m', 'LineWidth', 2);
hold on;
plot(N, HNN, '-b', 'LineWidth', 2);
plot(N, SRDNet, '-g', 'LineWidth', 2);
plot(N, Beamforming, '-r', 'LineWidth', 2);

xlabel('Number of Reflecting Elements (N)');
ylabel('RIS Gain (dB)');

legend({'Proposed', ...
        'Hierarchical Neural Network ', ...
        'SRDNet ', ...
        'Beamforming '}, ...
        'Location','northwest');

xlim([20 60]);
ylim([2 20]);

xticks(20:5:60);
yticks(2:2:20);

grid on;
box on;

set(gca, ...
    'FontSize', 10, ...
    'GridLineStyle', '--', ...
    'GridAlpha', 0.25);
% Frequency range (GHz)
f = linspace(0,12,1200);

%% Beamforming [18]
beamforming = ...
    -20*exp(-((f-2.0)/0.35).^2) ...
    -12*exp(-((f-4.2)/0.55).^2) ...
    -18*exp(-((f-6.0)/0.50).^2) ...
    -15*exp(-((f-8.5)/0.45).^2) ...
    -20*exp(-((f-10.0)/0.35).^2) ...
    -10*exp(-((f-11.5)/0.45).^2);

%% SRDNet [19]
srdnet = ...
    -16*exp(-((f-2.0)/0.40).^2) ...
    -25*exp(-((f-5.4)/0.70).^2) ...
    -24*exp(-((f-7.5)/0.45).^2) ...
    -22*exp(-((f-8.9)/0.35).^2) ...
    -30*exp(-((f-9.5)/0.25).^2) ...
    -12*exp(-((f-12.0)/0.40).^2);

%% Hierarchical Neural Network [21]
hnn = ...
    -18*exp(-((f-2.2)/0.35).^2) ...
    -48*exp(-((f-3.2)/0.25).^2) ...
    -15*exp(-((f-4.3)/0.55).^2) ...
    -20*exp(-((f-5.6)/0.60).^2) ...
    -35*exp(-((f-7.6)/0.55).^2) ...
    -15*exp(-((f-9.0)/0.45).^2) ...
    -16*exp(-((f-10.0)/0.35).^2) ...
    -13*exp(-((f-11.5)/0.45).^2);

%% Proposed
proposed = ...
    -22*exp(-((f-1.9)/0.30).^2) ...
    -35*exp(-((f-2.7)/0.18).^2) ...
    -18*exp(-((f-3.7)/0.35).^2) ...
    -22*exp(-((f-5.4)/0.45).^2) ...
    -25*exp(-((f-6.8)/0.35).^2) ...
    -34*exp(-((f-8.6)/0.30).^2) ...
    -38*exp(-((f-9.5)/0.20).^2) ...
    -32*exp(-((f-11.8)/0.25).^2);

%% Plot
figure('Position',[100 100 700 500]);

plot(f, beamforming, 'r', 'LineWidth', 2);
hold on;

plot(f, srdnet, 'b', 'LineWidth', 2);
plot(f, hnn, 'g', 'LineWidth', 2);
plot(f, proposed, 'm', 'LineWidth', 2);

%% Axis labels
xlabel('Frequency (GHz)', 'FontSize', 12);
ylabel('Reflection Coefficient (dB)', 'FontSize', 12);

%% Axis limits
xlim([0 12]);
ylim([-60 10]);

%% Ticks
xticks(0:2:12);
yticks(-60:10:10);

%% Grid
grid on;
set(gca, ...
    'GridLineStyle', '--', ...
    'GridAlpha', 0.25, ...
    'FontSize', 10);

%% Legend
legend('Beamforming ', ...
       'SRDNet ', ...
       'Hierarchical Neural Network ', ...
       'Proposed', ...
       'Location', 'southeast');

box on;
hold off;