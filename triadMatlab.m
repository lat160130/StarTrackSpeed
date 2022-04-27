mat = readmatrix("vectorInput.txt");
sb = mat(1,:);
si = mat(2,:);
mb = mat(3,:);
mi = mat(4,:);

t2b = cross(sb,mb)/ norm(cross(sb,mb));
t2i = cross(si,mi)/ norm(cross(si,mi));


t3b = cross(t2b, sb);
t3i = cross(t2i, si);

B = [sb' t2b' t3b'];
I = [si' t2i' t3i']';

R = B*I