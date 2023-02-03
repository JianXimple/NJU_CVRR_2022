function [bmap] = edgeGradient(im)
%UNTITLED 此处提供此函数的摘要
%   此处提供详细说明
    [mag,theta]=gradientMagnitude(im,1);
    tt=nonmax(mag.^0.7,theta);
    bmap=tt;
end

function[mag, theta] = gradientMagnitude(im, sigma)
    I=im;
    core = fspecial('gaussian',[5,5],sigma);
    I1=I(:,:,1);%提取红色分量
    I2=I(:,:,2);%提取绿色分量
    I3=I(:,:,3);%提取蓝色分量
    i1=imfilter(I1,core);
    i2=imfilter(I2,core);
    i3=imfilter(I3,core);
    res=cat(3,i1,i2,i3);
    % subplot(1,2,1);imshow(I),title('原图');
    % subplot(1,2,2);imshow(res),title('myself高斯滤波');
    r1=res(:,:,1);
    [g1,o1]=myf(r1);
    r2=res(:,:,2);
    [g2,o2]=myf(r2);
    r3=res(:,:,3);
    [g3,o3]=myf(r3);
    g=(g1.^2+g2.^2+g3.^3).^0.5;
    [r,c]=size(g);
    o=zeros(r,c);
    for i = 1:r
        for j = 1:c
            [m,index]=max([g1(i,j),g2(i,j),g3(i,j)]);
            if index==1
                o(i,j)=o1(i,j);
            elseif index==2
                o(i,j)=o2(i,j);
            else
                o(i,j)=o3(i,j);
            end
        end
    end
    mag=g;
    theta=o;
end
function [g,o]=myf(im)
    [fx,fy]=gradient(im);
    fx1=fx.^2;
    fy2=fy.^2;
    g=fx1+fy2;
    g=g.^0.5;
    o=atan(fy./fx);
end