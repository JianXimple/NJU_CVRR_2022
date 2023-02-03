I=imread('uttower1.jpg');
image(I) ;
I = single(rgb2gray(I)) ;
[f,d] = vl_sift(I) ;
perm = randperm(size(f,2)) ;
sel = perm(1:4) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ;
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;
h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
set(h3,'color','g') ;

Ia=imread('uttower1.jpg');
image(Ia) ;
Ia = single(rgb2gray(Ia)) ;
Ib=imread('uttower2.jpg');
image(Ib) ;
Ib = single(rgb2gray(Ib)) ;
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
[matches, scores] = vl_ubcmatch(da, db) ;

[sorted_array, sorted_index]=sort(scores);
id=sorted_index(end-3:end);
p1=[0,0];
p2=[0,0];
p3=[0,0];
p4=[0,0];
p_x=[];
p_y=[];
p0_x=[];
p0_y=[];
for i=1:4
    idx=id(i);
    m=matches(:,id);
    ffa=fa(:,m(1));
    ffb=fb(:,m(2));
    p_x(i)=ffa(1);
    p_y(i)=ffa(2);
    p0_x(i)=ffb(1);
    p0_y(i)=ffb(2);
end