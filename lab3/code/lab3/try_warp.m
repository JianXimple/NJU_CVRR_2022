im1=res_rgb(H,1);
im2=res_rgb(H,2);
im3=res_rgb(H,3);
[height1, width1] = size(im1);
res=zeros([height1,width1,3]);
res(:,:,1)=im1;
res(:,:,2)=im2;
res(:,:,3)=im3;
imshow(uint8(res));


function im=res_rgb(H,dim)
image1=imread('uttower1.jpg');
im1=image1(:,:,dim);
image2=imread('uttower2.jpg');
im2=image2(:,:,dim);
[height1, width1] = size(im1);
res=zeros([height1,2*width1]);
for i=1:height1
    for j=1:2*width1
        if j-width1>0
            res(i,j)=im1(i,j-width1);
        end
    end
end
res=uint8(res);
[height1, width1] = size(res);
[height2, width2] = size(im2);
im=res;
for i = 1:height1
        for j =1:width1
            im1_coord = [j-1024;i;1];
            transfered_coord = H*im1_coord;
%           im1_coord = [i-683 j];
%           transfered_coord=WarpH(im1_coord,H);
            %transfered_coord=[transfered_coord(2);transfered_coord(1);1];
            transfered_x = round(transfered_coord(1,1) / transfered_coord(3,1));
            transfered_y = round(transfered_coord(2,1) / transfered_coord(3,1));
            if transfered_x >= 0 &&transfered_x < width2 && transfered_y >= 0 && transfered_y < height2
                im(i,j) = im2(transfered_y+1,transfered_x+1);
                %disp(1);
            end
        end
end
%imshow(im);
end


function P2 = WarpH(P1, H)
x = P1(:, 1);
y = P1(:, 2);
p1 = [x'; y'; ones(1, length(x))];
q1 = H*p1;
q1 = q1./[q1(3, :); q1(3,:); q1(3, :)];
 
P2 = q1';
end