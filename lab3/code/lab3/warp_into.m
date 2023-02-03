HH=[[  8.34566914e-01  -3.12962592e-02  -4.53681006e+02];
   [  1.29611862e-01   1.21225212e+00  -5.04967813e+02];
   [ -8.20263106e-04   1.45634346e-05   1.00000000e+00]];
img2=imread("1.jpg");
img1=imread("4.jpg");
im1=img1(:,:,1);
im2=img2(:,:,1);
res=Warp(HH,im1,im2);
imshow(res);
function im=Warp(H,im1,im2)
    im=im1;
    [height1, width1] = size(im1);
    [height2, width2] = size(im2);
    for i = 1:height1
            for j =1:width1
                %disp(2);
                im1_coord = [j ;i ;1];
                transfered_coord = H*im1_coord;
                transfered_x = round(transfered_coord(1,1) / transfered_coord(3,1));
                transfered_y = round(transfered_coord(2,1) / transfered_coord(3,1));
                if transfered_x >= 0 &&transfered_x < width2 && transfered_y >= 0 && transfered_y < height2
                    im(i,j) = im2(transfered_y+1,transfered_x+1);
                    disp(1);
                end
            end
    end
end


