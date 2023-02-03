load filterBank.mat;
displayFilterBank(F);
i1=imread("coins.jpg");
i1=rgb2gray(i1);
i2=imread("planets.jpg");
i2=rgb2gray(i2);
i3=imread("snake.jpg");
i3=rgb2gray(i3);
imStack={i1,i2,i3};
for i=1:length(imStack)
    subplot(3,1,i);
    imshow(imStack{i});
end
textons= createTextons(imStack, F, 10);
% [featIm,labelIm]=extractTextonHists(i1,F,textons,5);
% imshow(label2rgb(labelIm));
[colorLabelIm, textureLabelIm] = compareSegmentations(imread("coins.jpg"), F,textons, 11, 10, 10);



function[labelIm] = quantizeFeats(featIm, meanFeats)
%featIm h*w*d 
%meanFeats k*d
%labelIm h*w
    [h,w,d]=size(featIm);
    [k,d]=size(meanFeats);
    temp=reshape(featIm,h*w,d);
    labelIm = kmeans(temp,k,'Start',meanFeats);
    labelIm=reshape(labelIm,h,w);
end
function[textons] = createTextons(imStack, bank, k)
%imStack n grayscale imgs
%bank m*m*d d filters
% k
% textons k*d
    [m,m,d]=size(bank);
    all_pixel=zeros(1000*length(imStack),d);
    for num=1:length(imStack)
        I=imStack{num};
        [temp_h,temp_w]=size(conv2(I,bank(:,:,1)));
%         disp(num);
%         disp(temp_h);
%         disp(temp_w);
        responses=zeros(temp_h-96,temp_w-96,38);
        for i=1:38
            responses(:,:,i)=conv2(I,bank(:,:,i),'valid');  
        end
        [r,c,d] = size(responses);
        t=reshape(responses,r*c,d);
        s = RandStream('mlfg6331_64'); 
        ti=randsample(s,r*c,1000);
        t=t(ti,:);
        %[idx,C]=kmeans(t,10);
        %all_pixel(:,:,num)=t;
        for t_idx=1:1000
            all_pixel(((num-1)*1000+t_idx),:)=t(t_idx,:);
        end
    end
    [idx,textons]=kmeans(all_pixel,k);
end


function[featIm,labelIm] = extractTextonHists(origIm, bank, textons, winSize)
    pad=(winSize-1)/2;
    pad=int64(pad);
    [m,m,d]=size(bank);
    for i=1:d
        responses(:,:,i)=conv2(origIm,bank(:,:,i),'valid');  
    end
    featIm=responses;
    labelIm=quantizeFeats(responses,textons);
    [h,w]=size(labelIm);
    disp(size(labelIm));
    ap=labelIm(1+pad:h-pad,1+pad:w-pad);
    [h1,w1]=size(ap);
end
function[colorLabelIm, textureLabelIm] = compareSegmentations(origIm, bank,textons, winSize, numColorRegions, numTextureRegions)
    textImg=rgb2gray(origIm);
    [featIm,textureLabelIm] = extractTextonHists(textImg, bank, textons, winSize);
    subplot(2,1,1);
    imshow(label2rgb(textureLabelIm));
    [h,w,d]=size(origIm);
    X = reshape(im2double(origIm), h*w, 3);
    colorLabelIm=kmeans(X,10);
    colorLabelIm=reshape(colorLabelIm,h,w);
    subplot(2,1,2);
    imshow(label2rgb(colorLabelIm));
end