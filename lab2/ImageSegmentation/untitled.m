origIm=imread("coins.jpg");
[h,w,d]=size(origIm);
X = reshape(im2double(origIm), h*w, 3);

colorLabelIm=kmeans(X,10);
colorLabelIm=reshape(colorLabelIm,h,w);
imshow(label2rgb(colorLabelIm));
