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
        for i=1:38
            responses(:,:,i)=conv2(rgb2gray(I),bank(:,:,i),'valid');  
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


function[featIm] = extractTextonHists(origIm, bank, textons, winSize)
    pad=(winSize-1)/2;
    [m,m,d]=size(bank);
    for i=1:d
        responses(:,:,i)=conv2(origIm,bank(:,:,i),'valid');  
    end
    featIm=responses;
    labelIm=quantizeFeats(responses,textons);
    [h,w]=size(labelIm);
    ap=labelIm(1+pad:h-pad,1+pad,w-pad);
    [h1,w1]=size(ap)
    for i=1:h1
        for j=1:w1
            %Iter padding
            subplot(h1,w1,(i-1)*w1+j);
            pix_n=ap(i-pad:i+pad,j-pad:j+pad);
            pix_n=pix_n(:);
            histogram(pix_n);
        end
    end
end

function[colorLabelIm, textureLabelIm] = compareSegmentations(origIm, bank,textons, winSize, numColorRegions, numTextureRegions)
    textImg=rgb2gray(origIm);
    textureLabelIm = extractTextonHists(textIm, bank, textons, winSize);


end