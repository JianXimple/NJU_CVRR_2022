clear
clc;
image1=imread('uttower1.jpg');
image2=imread('uttower2.jpg');
figure(1);
imshow(image1);
[xx,yy]=ginput(4);
figure(2);
imshow(image2);
[xa,ya]=ginput(4);
P1=[xx(1) yy(1);
    xx(2) yy(2);
    xx(3) yy(3);
    xx(4) yy(4)];
P2=[xa(1) ya(1);
    xa(2) ya(2);
    xa(3) ya(3);
    xa(4) ya(4)];
Point_H=[P1,P2];
 
H_Initial.img1=image1;
H_Initial.img2=image2;
H_Initial.Point=Point_H;
save H1.mat H_Initial;
