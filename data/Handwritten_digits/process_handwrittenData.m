
% load( 'E:\Program Files\MATLAB\R2015b\work\Data\Handwritten_digits\MINIST_source.mat'); 
% for i=0:9
%     a=1+100*i;
%    testimage0=reshape(Xs(:,a),[16 16]);%%%TEST
%    testimage0=testimage0';
%    imshow(testimage0,[]);%%%TEST
%    imwrite(mat2gray(testimage0),['MINIST_source',num2str(i),'.jpg']); %%%TEST  
% end

% load( 'E:\Program Files\MATLAB\R2015b\work\Data\Handwritten_digits\SEMEION_source.mat');
% testimage0=[];
% for i=0:9
%     a=1+100*i;
% %      testimage0=Xs{1};%%%TEST
%    testimage0=reshape(Xs{1}(:,a),[16 16]);%%%TEST
%    testimage0=testimage0';
%    imshow(testimage0,[]);%%%TEST
%    imwrite(mat2gray(testimage0),['SEMEION_source',num2str(i),'.jpg']); %%%TEST  
% end

load( 'E:\Program Files\MATLAB\R2015b\work\Data\Handwritten_digits\USPS_source.mat');
% testimage0=[];
for i=0:9
    a=1+100*i;
%      testimage0=Xs{1};%%%TEST
   testimage0=reshape(Xs{1}(:,a),[16 16]);%%%TEST
   testimage0=testimage0';
   imshow(testimage0,[]);%%%TEST
   imwrite(mat2gray(testimage0),['USPS_source',num2str(i),'.jpg']); %%%TEST  
end