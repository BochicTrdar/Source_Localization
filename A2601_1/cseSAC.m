function [Rf,ifile,ispt] = cseSAC(file,npfft,freqv,vchan,nsnap,noverlap,ifile,ispt);
%---------------------------------------------------------------------------
%  function cseSAC:  Compute estimated cross-spectral density matrices
%     from real baseband SACLANT array data stored in mat-files
%
%  [Rf,ifile,isnap] = cseSAC(file,npfft,freqv,vchan,nsnap,noverlap,
%                         ifile,ispt)
%
%  where npfft = length of snapshot in samples
%        freqv = normalized frequency vector (to sampling rate of 500 hz) 
%        vchan = vector of channel numbers to process 
%        file = basic file name of DREA mat-files
%        nsnap = number of snapshots to average
%        noverlap = number of samples to overlap snapshots
%        ifile = mat-file file number
%        ispt  = starting sample point in file 
%---------------------------------------------------------------------------

% where is ismax?????
Y = [];
j     =      sqrt(-1);
tpi   =          2*pi;
msens = length(vchan);
fmin  =      freqv(1);
nfreq = length(freqv);
ismin = 1; % Is this correct?
%
% define file parameters
%
fsr    =   1000;  %1000 hz. sampling rate
nchan  =     48;  %48  channels of data
lfils0 =  65536;  %length of file in samples/channel
lfils  = lfils0;
nrec   =      lfils/npfft;  %integer number of snapshots/file (>=1)
%ibin1  = (npfft*fmin) + 1;  %starting bin to process
ibin1  = fix(npfft*fmin) + 1;  %starting bin to process
wh     =   hamming(npfft);  %hamming window data segments 
%
% load first required data file
%
fnum  = sprintf('%g',ifile);
fname = [file,fnum];
disp(['Loading the SACLANT mat-file: ',fname]);
lcom = ['load ',fname];
eval(lcom);
dat = [dat(1:23,:);-dat(24,:);dat(25:48,:)].';%correct ch. 24 inversion & transpose
ldat = length(dat(:,1));  %length of data record
%
% set starting point if required to get signal or noise only data
% Note:  No checking done to make sure nsnap not greater than noise
% only or signal data!
%
if ispt == 0  %signal + noise case
  ispt = ismin;
elseif ispt == -1  %noise only desired
  if ismin > ldat-ismax
    ispt = 1;  %start before signal
  else
    ispt = ismax+1; %start after signal
  end
end
%
% start forming cross-spectral density matrices
%
Rf = zeros(msens*nfreq,msens);
x = zeros(npfft,msens);
for isnap = 1:nsnap
    iend = ispt + npfft - 1;  %index of last point in current snapshot
%
% load data file if required
%
  if iend > lfils
      dsav = dat(ispt:lfils,:);
      ifile = ifile + 1;
      fnum = sprintf('%g',ifile);
      fname = [file,fnum];
      disp(['Loading the SACLANT mat-file: ',fname]);
      lcom = ['load ',fname];
      eval(lcom);
      dat = [dat(1:23,:);-dat(24,:);dat(25:48,:)].';%correct ch. 24 inversion & transpose
      dat = [dsav;dat];  %prepend data from last file
      lfils = lfils0+lfils-ispt+1;
      ispt = 1;
  end
%
% take Hamming windowed fft's of each sensor output 
%
  for isens = 1:msens
    ichan = vchan(isens);
    ipt1 = ispt;
    ipt2 = ispt + npfft - 1;
    x(:,isens) = fft(wh.*dat(ipt1:ipt2,ichan));
   %x(:,isens) = fft(wh.*dat(ipt1:ipt2,ichan),npfft);
   %x(:,isens) = fft(dat(ipt1:ipt2,ichan),npfft);
  end
%
% form csdm estimate for frequency band of interest
%
  ibin = ibin1;
  for ib = 1:nfreq
    freq = fsr*(ibin-1)/npfft;
    y = (x(ibin,:)).';
    y = y/norm( y ); % Let me normalize...
%
% accumulate csdm estimate
%
    k1 = (ib-1)*msens+1;
    k2 = ib*msens;
    Rf(k1:k2,:) = Rf(k1:k2,:) + (y*y')/nsnap;
    ibin = ibin + 1;
  end
  ispt = ispt + npfft - noverlap;
end
%
%end of routine 
