or_bank=[0,30,45,60,90];
i=1;
for or =or_bank
        core1 = fspecial('gaussian',[5,1],1);
        core2 = fspecial('gaussian',[1,5],1);
        core=(cosd(or)*core1).*(sind(or)*core2);
        disp(core);
        subplot(1,5,i);
        imshow(double(mat2gray((core))));
        i=i+1;
end
