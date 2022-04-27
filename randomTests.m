
mat = readmatrix("vectorInput.txt");
fb = mat(1,:)';
fn = mat(2,:)';
mb = mat(3,:)';
mn = mat(4,:)';
Cbn = TRIAD(fb, mb, fn, mn)