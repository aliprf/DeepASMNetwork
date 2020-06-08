fid = fopen('annotation.txt', 'wt');
A = phisT;
for i=1:size(A,1)
   fprintf(fid, '%g\t', A(i,:));
   fprintf(fid, '\n');
end
for i=1:size(IsT,1)
    a = IsT{i,1};
    imwrite(a,strcat(num2str(i),'.png'))
end