function distance=Distl2(x,c)

  [d nx]=size(x);
  [d nc]=size(c);
  x2=sum(x.^2,1);
  c2=sum(c.^2,1);

  distance=sqrt(max(0,repmat(c2,nx,1)+repmat(x2',1,nc)-2*x'*c));
  
