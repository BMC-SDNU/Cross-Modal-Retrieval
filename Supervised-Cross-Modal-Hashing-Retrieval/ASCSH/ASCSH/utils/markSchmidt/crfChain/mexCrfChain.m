
% crfChain
fprintf('Compiling crfChain files...\n');
mex -outdir crfChain/mex crfChain/mex/crfChain_makePotentialsC.c
mex -outdir crfChain/mex crfChain/mex/crfChain_inferC.c
mex -outdir crfChain/mex crfChain/mex/crfChain_lossC2.c